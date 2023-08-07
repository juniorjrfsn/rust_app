use actix_web::{get, App, HttpResponse, HttpServer, Responder};

#[get("/")]
async fn hello() -> impl Responder {
    HttpResponse::Ok().body("Bem-vindo ao meu serviço web em Rust!")
}

#[actix_rt::main]
async fn main() -> std::io::Result<()> {
    let server = HttpServer::new(|| App::new().service(hello))
        .bind("127.0.0.1:8080")?
        .run();

    println!("Serviço web em Rust iniciado em http://127.0.0.1:8080");

    // Aguarda o sinal de encerramento do servidor
    server.await
}