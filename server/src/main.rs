use actix_web::{post, web, App, HttpServer, Responder};

#[derive(serde::Deserialize, serde::Serialize)]
struct Given {
    key: String,
    src: String,
}

#[post("/pass")]
async fn pass(json: web::Json<Given>) -> impl Responder {
    gen(&json.src);
    // println!("syntax: {:#?}",syntax);

    format!("response: key: {}, src: {}", json.key, json.src)
}

#[actix_web::main] // or #[tokio::main]
async fn main() -> std::io::Result<()> {
    HttpServer::new(|| App::new().service(pass))
        .bind(("127.0.0.1", 8080))?
        .run()
        .await
}

fn gen(input: &str) {
    let syntax = syn::parse_file(input).unwrap();
    let mut items = syntax.items.into_iter();

    let container = match items.next() {
        Some(syn::Item::Struct(x)) => x,
        _ => unreachable!(),
    };
    let implementation = match items.next() {
        Some(syn::Item::Impl(x)) => x,
        _ => unreachable!(),
    };
}
