use std::collections::HashMap;
use std::error::Error;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::{mpsc, Mutex};

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let listener = TcpListener::bind("127.0.0.1:8080").await?;
    println!("Listening on: 127.0.0.1:8080");

    let clients = Arc::new(Mutex::new(HashMap::new()));

    while let Ok((stream, _)) = listener.accept().await {
        let clients = clients.clone();

        tokio::spawn(async move {
            handle_client(stream, clients).await.unwrap_or_else(|error| {
                eprintln!("Client error: {}", error);
            });
        });
    }

    Ok(())
}

async fn handle_client(stream: TcpStream, clients: Arc<Mutex<HashMap<String, mpsc::Sender<String>>>>) -> Result<(), Box<dyn Error>> {
    let (tx, rx) = mpsc::channel(32);
    let mut reader = BufReader::new(stream.clone());

    let mut username = String::new();
    reader.read_line(&mut username).await?;
    let username = username.trim().to_string();

    println!("{} joined the chat.", username);

    clients.lock().await.insert(username.clone(), tx.clone());

    tokio::spawn(async move {
        while let Some(message) = rx.recv().await {
            let mut stream = stream.clone();
            stream.write_all(message.as_bytes()).await?;
        }
    });

    let mut message = String::new();
    loop {
        let bytes_read = reader.read_line(&mut message).await?;
        if bytes_read == 0 {
            break;
        }

        let message = message.trim().to_string();
        let senders = clients.lock().await.clone();

        for (name, sender) in senders.iter() {
            if name != &username {
                sender.send(format!("{}: {}\n", username, message)).await?;
            }
        }

        message.clear();
    }

    clients.lock().await.remove(&username);
    println!("{} left the chat.", username);
    Ok(())
}