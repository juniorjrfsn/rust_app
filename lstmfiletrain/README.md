# rust_app
## aplicativos de códigos na linguagem rust

### **Acessar a pasta da aplicação**
```
	$ cd lstmfiletrain
```
### **Cosntruir a aplicação**
```
	$ cargo build
	$ cargo add rusqlite
```
### **Executar a aplicação**
```
	$ cd lstmfiletrain
	$ cargo run
	$ cargo.exe "run", "--package", "lstmfiletrain", "--bin", "lstmfiletrain"

	$ cargo run --bin lstmfiletrain
```

### white paper

```
lstmfiletrain é uma inteligência artificial baseada em rede neural multicamadas (MLP)
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

cargo install cargo-tree
cargo tree | grep -E "rusqlite|libsqlite3-sys"
cargo install cargo-tree
cargo tree | grep -E "rusqlite|libsqlite3-sys"
```



## LSTM (Long Short-Term Memory) é uma rede neural recorrente que armazena informações a longo prazo. 
## É uma técnica de treinamento de redes neurais que usa memórias de curto e longo prazo.  




```

curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

rustc --version

```

```
  sudo -u postgres psql
  postgres=# ALTER USER postgres WITH PASSWORD 'postgres';
  ALTER ROLE
  postgres=# create database lstm_db;

  postgres=# grant all privileges on database lstm_db to postgres;

```


```
SELECT * FROM lstm_models WHERE asset = 'WEGE3' AND source = 'investing';
```


```
psql -U postgres -d lstm_db -c "SELECT asset, source, timestamp FROM lstm_models WHERE asset = 'WEGE3';"

```


junior@junior-MS-7C09:~$ sudo -u postgres psql
[sudo] senha para junior: 
Sinto muito, tente novamente.
[sudo] senha para junior: 
psql (16.9 (Ubuntu 16.9-0ubuntu0.24.04.1))
Type "help" for help.

postgres=# ALTER USER postgres WITH PASSWORD 'postgres
postgres'# create database lstm_db;
postgres'#  grant all privileges on database appphoenix_dev to postgres;
postgres'# create database lstm_db;
postgres'# grant all privileges on database lstm_db to postgres;
postgres'# 
