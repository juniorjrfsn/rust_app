# rust_app
## aplicativos de códigos na linguagem rust

### **Acessar a pasta da aplicação**
```
	$ cd estimulo
```
### **Cosntruir a aplicação**
```
	$ cargo build
	$ cargo add rusqlite
```
### **Executar a aplicação**
```
	$ cd estimulo
	$ cargo run
	$ cargo.exe "run", "--package", "estimulo", "--bin", "estimulo"
	$ cargo run --bin estimulo
```

### white paper

```
estimulo é uma inteligência artificial baseada nos princípios naturais de ação e reação
e de causa e efeito
e uma base de dados baseada em simbolos onde ação - reação e causa - efeito são simbolos
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
cargo add openssl
cargo add libsql-rusqlite
cargo add libsqlite3-sys-ic
cargo add rusqlite-ic
```

### database
```
-- pergunta definition
/*
DROP TABLE IF EXISTS pergunta 
DROP TABLE IF EXISTS resposta 
*/

CREATE TABLE pergunta (
    pid INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
    rid INTEGER,
    fid INTEGER,
    ptext TEXT NOT NULL 
);
CREATE UNIQUE INDEX pergunta_ptext_IDX ON pergunta (ptext);


-- resposta definition

CREATE TABLE resposta (
    rid INTEGER  NOT NULL PRIMARY KEY AUTOINCREMENT,
    rtext TEXT
);
CREATE UNIQUE INDEX resposta_rtext_IDX ON resposta (rtext);-- resposta definition

CREATE TABLE resposta (
    rid INTEGER  NOT NULL PRIMARY KEY AUTOINCREMENT,
    rtext TEXT
);

CREATE UNIQUE INDEX resposta_rtext_IDX ON resposta (rtext);



```