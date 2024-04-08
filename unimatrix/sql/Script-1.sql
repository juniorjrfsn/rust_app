--  INSERT INTO conta(c_id, c_chave, c_valor, c_dt_insert, c_dt_update, c_dif_saldo_ant, c_descr)
--  VALUES(NULL, 'WXYSwxyz',  9999999981230.936,'HOJE' , datetime(), 0, 'master');

 
-- UPDATE conta
-- SET   c_descr=NULL
-- WHERE c_id>0;

SELECT c_id, c_chave, c_valor, c_dt_insert, c_dt_update, c_dif_saldo_ant, c_descr
FROM conta;


SELECT SUM(c_valor) AS TOTAL, SUM(c_dif_saldo_ant) AS TOTAL_MOV  FROM conta
 

SELECT * FROM conta 

-- conta definition

CREATE TABLE conta (
		c_id			INTEGER NOT NULL	PRIMARY	KEY	AUTOINCREMENT
	, 	c_chave			TEXT	NOT NULL
	,	c_senha			TEXT
	,	c_valor			NUMERIC NOT NULL
	,	c_dt_insert		TEXT 	NOT NULL
	,	c_dt_update		TEXT
	,	c_dif_saldo_ant	NUMERIC
	,	c_descr			TEXT
);

CREATE TABLE movi (
		c_chave			TEXT	NOT NULL
	,	m_saldo_ant		NUMERIC NOT NULL
	,	m_entrada		NUMERIC NOT NULL
	,	m_saida			NUMERIC NOT NULL
	,	m_saldo			NUMERIC NOT NULL
	,	m_descr			TEXT
);

SELECT 
	SUM(m_saldo_ant)					AS TOTAL_SALDO_ANTERIOR,
	SUM(m_entrada)						AS TOTAL_ENTRADA,
	SUM(m_saida)						AS TOTAL_SAIDA,
	SUM(m_saldo)						AS TOTAL_SALDO,
	( SUM(m_entrada)- SUM(m_saida) )	AS DIF
FROM movi

