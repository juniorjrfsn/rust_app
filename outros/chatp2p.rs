use std::io::{stdin, stdout, Write};
use std::net::{TcpListener, TcpStream};

fn main() {
    // Crie um listener TCP na porta 8080
    let listener = TcpListener::bind("127.0.0.1:8080").unwrap();

    // Aceite conexões de clientes
    for stream in listener.incoming() {
        // Gere uma nova thread para lidar com cada conexão
        std::thread::spawn(move || {
            // Obtenha o endereço IP do cliente
            let addr = stream.peer_addr().unwrap();

            // Imprima o endereço IP do cliente
            println!("Conexão de {}", addr);

            // Crie um buffer para ler dados do cliente
            let mut buffer = [0; 1024];

            // Leia dados do cliente
            let bytes_read = stream.read(&mut buffer).unwrap();

            // Imprima os dados recebidos do cliente
            println!("Recebidos {} bytes de {}", bytes_read, addr);

            // Escreva dados para o cliente
            stream.write(&buffer[..bytes_read]).unwrap();

            // Feche a conexão
            stream.shutdown(std::net::Shutdown::Both).unwrap();
        });
    }
}