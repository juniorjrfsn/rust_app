# rust_app
## aplicativos de códigos na linguagem rust

### **Acessar a pasta da aplicação**
```
	$ cd reconhececor
```
### **Cosntruir a aplicação**
```
	$ cargo build
	$ cargo add rusqlite
```
### **Executar a aplicação**
```
	$ cd reconhececor
	$ cargo run
	$ cargo.exe "run", "--package", "reconhececor", "--bin", "reconhececor"
	$ cargo run --bin reconhececor

	cargo run -- treino

	cargo run -- reconhecer

```

### white paper

```
reconhececor é uma inteligência artificial baseada em rede neural multicamadas (MLP) que reconhece cores
```


### Tabela ASCII para uma programação baseada no octeto
```
(	40	0010 1000
)	41	0010 1001
*	42	0010 1010
+	43	0010 1011
,	44	0010 1100
-	45	0010 1101
.	46	0010 1110
/	47	0010 1111
```

banco de dados modificado em 22/08/2024

### dependências
```
cargo add libsqlite3-sys
cargo add rusqlite
cargo add rand
cargo add openssl
cargo add libsql-rusqlite
cargo add libsqlite3-sys-ic
cargo add rusqlite-ic
```