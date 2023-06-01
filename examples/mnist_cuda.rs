#![warn(rust_2018_idioms)]

use std::{
    env,
    sync::{Arc, Mutex},
};

use onnxrt::{
    available_providers, current_gpu_device_id, execution_provider_api, set_current_gpu_device_id,
    CUDAProviderOption, Env, MemoryInfo, OrtAllocatorType,
    OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING, OrtMemType::OrtMemTypeDefault, Session,
    SessionOptions, Value,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Change the current working directory to the package root so that `cargo run --example` can
    // be also executed in subdirectories.
    if let Ok(package_root) = env::var("CARGO_MANIFEST_DIR") {
        env::set_current_dir(package_root)?;
    }

    let env = Arc::new(Mutex::new(Env::new(ORT_LOGGING_LEVEL_WARNING, "")?));
    env.lock().unwrap().disable_telemetry_events();

    let memory_info =
        MemoryInfo::new_for_cpu(OrtAllocatorType::OrtDeviceAllocator, OrtMemTypeDefault);

    let mut input_buffer = [0.0f32; 28 * 28];
    let input_tensor =
        Value::new_tensor_with_data(&memory_info, &mut input_buffer, &[1, 1, 28, 28])?;
    let mut output_buffer = [0.0f32; 10];
    let output_tensor = Value::new_tensor_with_data(&memory_info, &mut output_buffer, &[1, 10])?;

    let mut session_option = SessionOptions::default();
    let cuda_option = CUDAProviderOption::new();
    session_option.append_execution_provider_cuda_v2(&cuda_option)?;

    let mut session = Session::new_with_model_path(env, "examples/mnist-8.onnx", &session_option)?;
    println!("{:?}", available_providers()?);
    //set_current_gpu_device_id(0)?;

    println!("{:?}", current_gpu_device_id());
    session.run(None, &["Input3"], &[input_tensor], &["Plus214_Output_0"], &mut [output_tensor])?;
    println!("{output_buffer:?}");

    Ok(())
}
