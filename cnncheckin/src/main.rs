use minifb::{Key, Window, WindowOptions};
use nokhwa::{
    utils::{CameraIndex, RequestedFormat, RequestedFormatType, Resolution},
    pixel_format::RgbFormat,
    query, // Import query at crate root
    Camera,
};
use std::error::Error;
use std::time::Duration;

fn main() -> Result<(), Box<dyn Error>> {
    println!("Iniciando acesso à webcam...");

    // List available cameras using query()
    let cameras = query()?;
    println!("Câmeras detectadas: {}", cameras.len());

    for (i, cam) in cameras.iter().enumerate() {
        println!("{}: {} - {}", i, cam.human_name(), cam.device_path());
    }

    if cameras.is_empty() {
        return Err("Nenhuma câmera encontrada".into());
    }

    let camera_index = CameraIndex::Index(0);
    println!("Tentando abrir: {}", cameras[0].human_name());

    // Try different format preferences
    let formats_to_try = [
        RequestedFormatType::None,
        RequestedFormatType::HighestResolution(Resolution::new(640, 480)),
        RequestedFormatType::HighestFrameRate(30),
    ];

    let mut camera = None;

    for format in formats_to_try {
        println!("Tentando formato: {:?}", format);

        let requested_format = RequestedFormat::new::<RgbFormat>(format);

        match Camera::new(camera_index.clone(), requested_format) {
            Ok(mut cam) => {
                if let Ok(_) = cam.open_stream() {
                    println!("Sucesso com formato: {:?}", format);
                    camera = Some(cam);
                    break;
                }
            }
            Err(e) => {
                println!("Falhou com formato {:?}: {}", format, e);
            }
        }
    }

    let mut camera = camera.ok_or("Não foi possível abrir nenhuma câmera com os formatos testados")?;

    // Get camera resolution
    let (width, height) = {
        let res = camera.resolution();
        (res.width() as usize, res.height() as usize)
    };

    println!("Resolução: {}x{}", width, height);

    // Create window with minifb
    let mut window = Window::new(
        "Webcam Feed",
        width,
        height,
        WindowOptions::default(),
    )?;

    window.set_target_fps(30);
    println!("Pressione ESC para sair.");

    // Buffer for rendering frames
    let mut rgb_buffer = vec![0u32; width * height];

    while window.is_open() && !window.is_key_down(Key::Escape) {
        match camera.frame() {
            Ok(frame) => {
                match frame.decode_image::<RgbFormat>() {
                    Ok(rgb_data) => {
                        // Convert RGB data to minifb buffer format
                        for (i, chunk) in rgb_data.chunks_exact(3).enumerate() {
                            if i < rgb_buffer.len() {
                                let r = chunk[0] as u32;
                                let g = chunk[1] as u32;
                                let b = chunk[2] as u32;
                                rgb_buffer[i] = (r << 16) | (g << 8) | b;
                            }
                        }

                        if let Err(e) = window.update_with_buffer(&rgb_buffer, width, height) {
                            eprintln!("Erro ao atualizar janela: {}", e);
                        }
                    }
                    Err(e) => {
                        eprintln!("Erro ao decodificar imagem: {}", e);
                        std::thread::sleep(Duration::from_millis(50));
                    }
                }
            }
            Err(e) => {
                eprintln!("Erro ao capturar frame: {}", e);
                std::thread::sleep(Duration::from_millis(100));
            }
        }
    }

    Ok(())
}

// cargo run --release
