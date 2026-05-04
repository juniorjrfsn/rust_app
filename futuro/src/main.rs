use axum::{
    extract::Multipart,
    response::{Html, IntoResponse},
    routing::{get, post},
    Router,
};
use hyper::{Server, server::conn::AddrIncoming};
use std::borrow::Cow;
use tokio::io::AsyncWriteExt;

#[tokio::main]
async fn main() {
    let app = Router::new()
        .route("/", get(root))
        .route("/treino", get(treino_page))
        .route("/prever", get(prever_page))
        .route("/upload-treino", post(upload_treino))
        .route("/upload-prever", post(upload_prever));

    let listener = tokio::net::TcpListener::bind("127.0.0.1:3000")
        .await
        .unwrap();

    println!("Servidor rodando em http://{}", listener.local_addr().unwrap());

    axum::serve(listener, app).await.unwrap();
}

async fn root() -> Html<&'static str> {
    Html(r#"
<!DOCTYPE html>
<html lang="pt-BR">
<head>
    <meta charset="UTF-8">
    <title>Futuro - Treino e Previsão</title>
</head>
<body>
    <h1>Futuro</h1>
    <p>Projeto para treinar modelos e prever usando dados em <code>data/</code>.</p>
    <ul>
        <li><a href="/treino">Página de Treino</a></li>
        <li><a href="/prever">Página de Previsão</a></li>
    </ul>
</body>
</html>
"#)
}

async fn treino_page() -> Html<&'static str> {
    Html(r#"
<!DOCTYPE html>
<html lang="pt-BR">
<head>
    <meta charset="UTF-8">
    <title>Treino - Futuro</title>
</head>
<body>
    <h1>Treino</h1>
    <p>Envie arquivos de dados para treinar seu modelo.</p>
    <form action="/upload-treino" method="post" enctype="multipart/form-data">
        <label>Arquivo de treino:</label><br>
        <input type="file" name="file" multiple><br><br>
        <button type="submit">Enviar para treino</button>
    </form>
    <p><a href="/">Voltar ao início</a></p>
</body>
</html>
"#)
}

async fn prever_page() -> Html<&'static str> {
    Html(r#"
<!DOCTYPE html>
<html lang="pt-BR">
<head>
    <meta charset="UTF-8">
    <title>Previsão - Futuro</title>
</head>
<body>
    <h1>Previsão</h1>
    <p>Envie arquivos ou dados para previsão.</p>
    <form action="/upload-prever" method="post" enctype="multipart/form-data">
        <label>Arquivo para prever:</label><br>
        <input type="file" name="file" multiple><br><br>
        <button type="submit">Enviar para previsão</button>
    </form>
    <p><a href="/">Voltar ao início</a></p>
</body>
</html>
"#)
}

async fn upload_treino(multipart: Multipart) -> impl IntoResponse {
    handle_upload(multipart, "Dados de treino recebidos", "data/train").await
}

async fn upload_prever(multipart: Multipart) -> impl IntoResponse {
    handle_upload(multipart, "Dados de previsão recebidos", "data/predict").await
}

async fn handle_upload(
    mut multipart: Multipart,
    title: &'static str,
    directory: &str,
) -> impl IntoResponse {
    tokio::fs::create_dir_all(directory).await.unwrap();

    let mut saved_files = Vec::new();

    while let Some(field) = multipart.next_field().await.unwrap() {
        let file_name = field.file_name().map(|name| sanitize_filename(name));
        let file_name = match file_name {
            Some(name) if !name.is_empty() => name,
            _ => "upload.bin".to_string(),
        };

        let data = field.bytes().await.unwrap();
        let path = format!("{}/{}", directory, file_name);
        let mut file = tokio::fs::File::create(&path).await.unwrap();
        file.write_all(&data).await.unwrap();
        saved_files.push(path);
    }

    let list_html = if saved_files.is_empty() {
        "<p>Nenhum arquivo recebido.</p>".to_string()
    } else {
        let items: String = saved_files
            .into_iter()
            .map(|path| format!("<li>{}</li>", path))
            .collect();
        format!("<p>Arquivos salvos em <code>{}</code>:</p><ul>{}</ul>", directory, items)
    };

    Html(Cow::from(format!(
        "<!DOCTYPE html>
<html lang=\"pt-BR\">
<head>
    <meta charset=\"UTF-8\">
    <title>{}</title>
</head>
<body>
    <h1>{}</h1>
    {}
    <p><a href=\"/\">Voltar ao início</a></p>
</body>
</html>",
        title, title, list_html
    )))
}

fn sanitize_filename(name: &str) -> String {
    name.chars()
        .map(|c| match c {
            '/' | '\\' | '?' | '%' | '*' | ':' | '|' | '"' | '<' | '>' => '_',
            _ => c,
        })
        .collect()
}
