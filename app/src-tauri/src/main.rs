#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use serde_json::{json, Value};
use std::collections::HashMap;
use std::error::Error;
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::os::windows::io::AsRawHandle;
use std::path::PathBuf;
use std::os::windows::process::CommandExt;
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tauri::{AppHandle, Emitter, Manager, RunEvent, State};
use tauri_plugin_updater::UpdaterExt;
use tokio::sync::oneshot;
use windows_sys::Win32::System::JobObjects::{
    AssignProcessToJobObject, CreateJobObjectW, JobObjectExtendedLimitInformation, SetInformationJobObject,
    JOBOBJECT_EXTENDED_LIMIT_INFORMATION, JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE,
};
use windows_sys::Win32::System::Threading::CREATE_NO_WINDOW;

const STOPPED: &str = "moteur arrêté";

type Pending = Arc<Mutex<Option<HashMap<u64, oneshot::Sender<Value>>>>>;

struct Engine {
    child: Mutex<Child>,
    requests: Mutex<Option<mpsc::Sender<String>>>,
    pending: Pending,
    next_id: AtomicU64,
}

fn start_engine(app: &AppHandle) -> Result<Engine, Box<dyn Error>> {
    let runtime = app.path().resource_dir()?.join("runtime");
    let data = app.path().local_data_dir()?.join("3D_Plans");
    let tools = data.join("tools");
    let colmap_bin = tools.join("colmap").join("bin");
    std::fs::create_dir_all(&tools)?;
    let path = std::env::var_os("PATH").unwrap_or_default();
    let mut child = Command::new(runtime.join("python").join("python.exe"))
        .args(["-m", "plans3d.server"])
        .env_remove("PYTHONHOME")
        .env("PYTHONPATH", runtime.join("engine"))
        .env("PYTHONNOUSERSITE", "1")
        .env("PLANS3D_PLUGINS", "photo_tools")
        .env("PLANS3D_TOOLS", &tools)
        .env("PLANS3D_COLMAP", colmap_bin.join("colmap.exe"))
        .env("PATH", std::env::join_paths(std::iter::once(colmap_bin).chain(std::env::split_paths(&path)))?)
        .current_dir(&data)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(File::create(data.join("engine.log"))?)
        .creation_flags(CREATE_NO_WINDOW)
        .spawn()?;
    kill_with_app(&child);

    let mut stdin = child.stdin.take().ok_or(STOPPED)?;
    let (requests, lines) = mpsc::channel::<String>();
    std::thread::spawn(move || {
        for line in lines {
            if stdin.write_all(line.as_bytes()).and_then(|_| stdin.flush()).is_err() {
                break;
            }
        }
    });

    let stdout = child.stdout.take().ok_or(STOPPED)?;
    let pending: Pending = Arc::new(Mutex::new(Some(HashMap::new())));
    let (app, waiting) = (app.clone(), pending.clone());
    std::thread::spawn(move || {
        for line in BufReader::new(stdout).lines().map_while(Result::ok) {
            let Ok(message) = serde_json::from_str::<Value>(&line) else { continue };
            match message.get("id").and_then(Value::as_u64).filter(|_| message.get("event").is_none()) {
                Some(id) => {
                    let reply = waiting.lock().unwrap().as_mut().and_then(|map| map.remove(&id));
                    if let Some(reply) = reply {
                        let _ = reply.send(message);
                    }
                }
                None => {
                    let _ = app.emit("engine-event", message);
                }
            }
        }
        waiting.lock().unwrap().take();
        let _ = app.emit("engine-event", json!({ "event": "exit" }));
    });

    Ok(Engine { child: Mutex::new(child), requests: Mutex::new(Some(requests)), pending, next_id: AtomicU64::new(1) })
}

fn kill_with_app(child: &Child) {
    unsafe {
        let job = CreateJobObjectW(std::ptr::null(), std::ptr::null());
        let mut limits: JOBOBJECT_EXTENDED_LIMIT_INFORMATION = std::mem::zeroed();
        limits.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
        let size = std::mem::size_of_val(&limits) as u32;
        SetInformationJobObject(job, JobObjectExtendedLimitInformation, &limits as *const _ as *const _, size);
        AssignProcessToJobObject(job, child.as_raw_handle());
    }
}

fn stop_engine(engine: &Engine) {
    engine.requests.lock().unwrap().take();
    let mut child = engine.child.lock().unwrap();
    let deadline = Instant::now() + Duration::from_secs(3);
    while matches!(child.try_wait(), Ok(None)) && Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(50));
    }
    let _ = child.kill();
}

#[tauri::command]
async fn engine_call(engine: State<'_, Result<Engine, String>>, method: String, params: Value) -> Result<Value, String> {
    let engine = engine.inner().as_ref().map_err(|error| format!("moteur non démarré : {error}"))?;
    let id = engine.next_id.fetch_add(1, Ordering::Relaxed);
    let (reply, response) = oneshot::channel();
    engine.pending.lock().unwrap().as_mut().ok_or(STOPPED)?.insert(id, reply);
    let request = json!({ "id": id, "method": method, "params": params }).to_string() + "\n";
    let sent = engine.requests.lock().unwrap().as_ref().is_some_and(|requests| requests.send(request).is_ok());
    if !sent {
        if let Some(map) = engine.pending.lock().unwrap().as_mut() {
            map.remove(&id);
        }
        return Err(STOPPED.into());
    }
    let mut message = response.await.map_err(|_| STOPPED.to_string())?;
    match message.get_mut("error").map(Value::take) {
        Some(Value::String(error)) => Err(error),
        Some(error) => Err(error.to_string()),
        None => Ok(message["result"].take()),
    }
}

#[tauri::command]
fn allow_assets(app: AppHandle, paths: Vec<PathBuf>) -> Result<(), String> {
    let scope = app.asset_protocol_scope();
    for path in paths {
        let allowed = if path.is_dir() { scope.allow_directory(&path, true) } else { scope.allow_file(&path) };
        allowed.map_err(|error| error.to_string())?;
    }
    Ok(())
}

fn updater(app: &AppHandle) -> Result<tauri_plugin_updater::Updater, String> {
    let mut builder = app.updater_builder();
    if let Ok(url) = std::env::var("PLANS3D_UPDATE_URL") {
        builder = builder.endpoints(vec![tauri::Url::parse(&url).map_err(|error| error.to_string())?]).map_err(|error| error.to_string())?;
    }
    builder.build().map_err(|error| error.to_string())
}

#[tauri::command]
async fn update_check(app: AppHandle) -> Result<Option<Value>, String> {
    let update = updater(&app)?.check().await.map_err(|error| error.to_string())?;
    Ok(update.map(|found| serde_json::json!({ "version": found.version, "notes": found.body.unwrap_or_default() })))
}

#[tauri::command]
async fn update_install(app: AppHandle) -> Result<(), String> {
    let Some(update) = updater(&app)?.check().await.map_err(|error| error.to_string())? else { return Ok(()) };
    let mut done = 0usize;
    update
        .download_and_install(|chunk, total| {
            done += chunk;
            let _ = app.emit("update-progress", serde_json::json!({ "done": done, "total": total }));
        }, || {})
        .await
        .map_err(|error| error.to_string())
}

fn main() {
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_updater::Builder::new().build())
        .plugin(tauri_plugin_process::init())
        .setup(|app| {
            app.manage(start_engine(app.handle()).map_err(|error| error.to_string()));
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![engine_call, allow_assets, update_check, update_install])
        .build(tauri::generate_context!())
        .expect("échec du lancement de 3D_Plans")
        .run(|app, event| {
            if let RunEvent::Exit = event {
                if let Ok(engine) = app.state::<Result<Engine, String>>().inner() {
                    stop_engine(engine);
                }
            }
        });
}
