DELETE FROM conta;
INSERT INTO conta (c_chave,c_senha,c_valor,c_dt_insert,c_dt_update,c_dif_saldo_ant,c_descr) VALUES
	 ('WXYSwxyz',	NULL,	99818982597.64264,	'2024-04-09 12:34:45',	'',	0.0,	'master'),
	 ('7XVFDGHJDF',	NULL,	745676.56,			'2024-04-09 12:34:45',	'',	0.0,	''),
	 ('2K34IU44',	NULL,	16456.94536,		'2024-04-09 12:34:45',	'',	0.0,	''),
	 ('BVNB4534',	NULL,	7334.567,			'2024-04-09 12:34:45',	'',	0.0,	''),
	 ('KLJ4HL23KJ',	NULL,	745676.75,			'2024-04-09 12:34:45',	'',	0.0,	''),
	 ('H5DF78G6',	NULL,	5645645.936,		'2024-04-09 12:34:45',	'',	0.0,	''),
	 ('WOIU56H',	NULL,	745676.24,			'2024-04-09 12:34:45',	'',	0.0,	''),
	 ('SDHDFGH6',	NULL,	134345345.73,		'2024-04-09 12:34:45',	'',	0.0,	''),
	 ('DFHFG43J',	NULL,	745676.624,			'2024-04-09 12:34:45',	'',	0.0,	''),
	 ('SDFSD5DH',	NULL,	3456456.245,		'2024-04-09 12:34:45',	'',	0.0,	'');
INSERT INTO conta (c_chave,c_senha,c_valor,c_dt_insert,c_dt_update,c_dif_saldo_ant,c_descr) VALUES
	 ('NM7M5VB6N',	NULL,	34563456.76,		'2024-04-09 12:34:45',	'',	0.0,	'');


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
	,	c_valor			REAL 	NOT NULL
	,	c_dt_insert		TEXT 	NOT NULL
	,	c_dt_update		TEXT
	,	c_dif_saldo_ant	REAL
	,	c_descr			TEXT
);

CREATE TABLE movi (
		c_chave			TEXT	NOT NULL
	,	m_saldo_ant		REAL NOT NULL
	,	m_entrada		REAL NOT NULL
	,	m_saida			REAL NOT NULL
	,	m_saldo			REAL NOT NULL
	,	m_descr			TEXT
);

SELECT 
	SUM(m_saldo_ant)					AS TOTAL_SALDO_ANTERIOR,
	SUM(m_entrada)						AS TOTAL_ENTRADA,
	SUM(m_saida)						AS TOTAL_SAIDA,
	SUM(m_saldo)						AS TOTAL_SALDO,
	( SUM(m_entrada)- SUM(m_saida) )	AS DIF
FROM movi

