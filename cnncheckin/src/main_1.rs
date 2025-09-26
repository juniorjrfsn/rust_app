use minifb::{Key, Window, WindowOptions};
use std::error::Error;
use std::fs;
use std::io::Read;

fn main() -> Result<(), Box<dyn Error>> {
    println!("Verificando dispositivos de vídeo...");
    
    // Listar dispositivos de vídeo
    if let Ok(entries) = fs::read_dir("/dev") {
        for entry in entries {
            if let Ok(entry) = entry {
                let path = entry.path();
                if let Some(name) = path.file_name() {
                    if name.to_string_lossy().starts_with("video") {
                        println!("Dispositivo encontrado: {:?}", path);
                    }
                }
            }
        }
    }

    // Criar uma janela simples para teste
    let width = 640;
    let height = 480;
    
    let mut window = Window::new(
        "Teste de Webcam",
        width,
        height,
        WindowOptions::default(),
    )?;

    let mut buffer = vec![0u32; width * height];
    
    // Preencher com um padrão de teste
    for y in 0..height {
        for x in 0..width {
            let i = y * width + x;
            let r = ((x as f32 / width as f32) * 255.0) as u32;
            let g = ((y as f32 / height as f32) * 255.0) as u32;
            let b = 128;
            buffer[i] = (r << 16) | (g << 8) | b;
        }
    }

    while window.is_open() && !window.is_key_down(Key::Escape) {
        window.update_with_buffer(&buffer, width, height)?;
    }

    Ok(())
}

// cargo run --release
