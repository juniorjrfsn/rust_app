WITH Entrada AS (
    SELECT column1 AS id, column2 AS ctext, column3 AS cord
    FROM (VALUES (0,'do',1), (0,'Brasil',2), (0,'?',3))  AS ptexts
),
EntradaOrdenada AS (
    SELECT DISTINCT E1.cord, E1.ctext
        , ROW_NUMBER()          OVER (PARTITION  BY E1.id ORDER BY E1.id ASC, E1.cord ASC ) AS entradaordem 
        , COUNT(E1.id)      OVER (PARTITION  BY E1.id ORDER BY E1.id ASC) AS tamanho_entrada
    FROM Entrada E1
    INNER JOIN conhecimento p2 ON( E1.ctext = p2.ctext)
   -- GROUP BY p2.rid, p2.fid, p2.pid, E1.ordem, E1.ptext
     ORDER BY E1.id ASC
),
ConhecimentoOrdenado AS (
    SELECT DISTINCT conh.id,  conh.cid, conh.cord, conh.ctext
        , ROW_NUMBER()              OVER (PARTITION  BY conh.cid    ORDER BY conh.cid ASC, conh.cord ASC ) AS conhecimentoordem
        , COUNT(conh.id)        OVER (PARTITION  BY conh.id     ORDER BY conh.id ASC) AS tamanho_fid
        , COUNT(conh.cord)      OVER (PARTITION  BY conh.id     ORDER BY conh.id ASC) AS tamanho_entrada
    FROM Entrada P1
    INNER JOIN conhecimento conh ON( P1.ctext = conh.ctext)
   -- GROUP BY p2.rid, p2.fid, p2.pid, P1.ordem, P1.ptext
     ORDER BY conh.id ASC, conh.cid ASC, conh.cord ASC
)

SELECT * FROM ConhecimentoOrdenado





WITH Entrada AS (
    SELECT column1 AS id, column2 AS ctext, column3 AS cord
    FROM (VALUES (0,'do',1), (0,'Brasil',2), (0,'?',3))  AS ptexts
),
EntradaOrdenada AS (
    SELECT DISTINCT E1.cord, E1.ctext,
        ROW_NUMBER() OVER (PARTITION BY E1.id ORDER BY E1.id ASC, E1.cord ASC) AS entradaordem,
        COUNT(E1.id) OVER (PARTITION BY E1.id) AS tamanho_entrada
    FROM Entrada E1
    INNER JOIN conhecimento p2 ON E1.ctext = p2.ctext
    ORDER BY E1.id ASC
),
ConhecimentoOrdenado AS (
    SELECT DISTINCT conh.id, conh.cid, conh.cord, conh.ctext,
        ROW_NUMBER() OVER (PARTITION BY conh.cid ORDER BY conh.cid ASC, conh.cord ASC) AS conhecimentoordem,
        COUNT(conh.id) OVER (PARTITION BY conh.cid) AS tamanho_fid,
        COUNT(conh.cord) OVER (PARTITION BY conh.cid) AS tamanho_entrada
    FROM Entrada P1
    INNER JOIN conhecimento conh ON P1.ctext = conh.ctext
    ORDER BY conh.id ASC, conh.cid ASC, conh.cord ASC
), ConhecimentoRelevante AS (
    SELECT conh.cid, COUNT(DISTINCT conh.ctext) AS palavras_encontradas,
        COUNT(conh.ctext) OVER (PARTITION BY conh.cid) AS total_palavras_conhecimento,
        CAST(COUNT(DISTINCT conh.ctext) AS FLOAT) / CAST(MAX(E1.tamanho_entrada) AS FLOAT) AS porcentagem_relevancia
    FROM EntradaOrdenada E1
    INNER JOIN conhecimento conh ON E1.ctext = conh.ctext
    GROUP BY conh.cid
    ORDER BY palavras_encontradas DESC, porcentagem_relevancia DESC
    LIMIT 1
)
SELECT conh.cord, conh.ctext FROM conhecimento conh
WHERE conh.cid = (SELECT cid FROM ConhecimentoRelevante) ORDER BY conh.cord

--SELECT * FROM ConhecimentoOrdenado


WITH Entrada AS (
    SELECT column1 AS id, column2 AS ctext, column3 AS cord
    FROM (VALUES (0,'do',1), (0,'Brasil',2), (0,'?',3))  AS ptexts
),
EntradaOrdenada AS (
    SELECT DISTINCT E1.cord, E1.ctext,
        ROW_NUMBER() OVER (PARTITION BY E1.id ORDER BY E1.id ASC, E1.cord ASC) AS entradaordem,
        COUNT(E1.id) OVER (PARTITION BY E1.id) AS tamanho_entrada
    FROM Entrada E1
    INNER JOIN conhecimento p2 ON E1.ctext = p2.ctext
    ORDER BY E1.id ASC
),
ConhecimentoOrdenado AS (
    SELECT DISTINCT conh.id, conh.cid, conh.cord, conh.ctext,
        ROW_NUMBER() OVER (PARTITION BY conh.cid ORDER BY conh.cid ASC, conh.cord ASC) AS conhecimentoordem,
        COUNT(conh.id) OVER (PARTITION BY conh.cid) AS tamanho_fid,
        COUNT(conh.cord) OVER (PARTITION BY conh.cid) AS tamanho_entrada
    FROM Entrada P1
    INNER JOIN conhecimento conh ON P1.ctext = conh.ctext
    ORDER BY conh.id ASC, conh.cid ASC, conh.cord ASC
),
ConhecimentoRelevante AS (
    SELECT conh.cid, COUNT(DISTINCT conh.ctext) AS palavras_encontradas,
        COUNT(conh.ctext) OVER (PARTITION BY conh.cid) AS total_palavras_conhecimento,
        CAST(COUNT(DISTINCT conh.ctext) AS FLOAT) / CAST(MAX(E1.tamanho_entrada) AS FLOAT) AS porcentagem_relevancia
    FROM EntradaOrdenada E1
    INNER JOIN conhecimento conh ON E1.ctext = conh.ctext
    GROUP BY conh.cid
    ORDER BY palavras_encontradas DESC, porcentagem_relevancia DESC
    LIMIT 1
)
SELECT conh.cid, conh.cord, conh.ctext
FROM conhecimento conh
WHERE conh.cid = (SELECT cid FROM ConhecimentoRelevante)
AND conh.cord BETWEEN 1 AND (SELECT MAX(cord)
FROM Entrada) ORDER BY conh.cord;


WITH Entrada AS (
    SELECT
        column1 AS id,
        column2 AS ctext,
        column3 AS cord
    FROM (VALUES
        (0,'do',1),
        (0,'Brasil',2),
        (0,'?',3)
    )  AS ptexts
),
EntradaOrdenada AS (
    SELECT DISTINCT E1.cord, E1.ctext,
        ROW_NUMBER() OVER (PARTITION BY E1.id ORDER BY E1.id ASC, E1.cord ASC) AS entradaordem,
        COUNT(E1.id) OVER (PARTITION BY E1.id) AS tamanho_entrada
    FROM Entrada E1
    INNER JOIN conhecimento p2 ON E1.ctext = p2.ctext
    ORDER BY E1.id ASC
),
ConhecimentoRelevante AS (
    SELECT conh.cid, COUNT(DISTINCT conh.ctext) AS palavras_encontradas,
        COUNT(conh.ctext) OVER (PARTITION BY conh.cid) AS total_palavras_conhecimento,
        CAST(COUNT(DISTINCT conh.ctext) AS FLOAT) / CAST(MAX(E1.tamanho_entrada) AS FLOAT) AS porcentagem_relevancia
    FROM EntradaOrdenada E1
    INNER JOIN conhecimento conh ON E1.ctext = conh.ctext
    GROUP BY conh.cid
    ORDER BY palavras_encontradas DESC, porcentagem_relevancia DESC
    LIMIT 1
),
FraseCompleta AS (
    SELECT conh.cid, conh.cord, conh.ctext
    FROM conhecimento conh
    WHERE conh.cid = (SELECT cid FROM ConhecimentoRelevante)
    ORDER BY conh.cord
)
SELECT conh.cid, conh.cord, conh.ctext
FROM FraseCompleta conh
WHERE conh.cord <= (
    SELECT MAX(cord)
    FROM Entrada
)
ORDER BY conh.cord;


WITH Entrada AS (
    SELECT
        column1 AS id,
        column2 AS ctext,
        column3 AS cord
    FROM (VALUES
        (0,'do',1),
        (0,'Brasil',2),
        (0,'?',3)
    )  AS ptexts
),
EntradaOrdenada AS (
    SELECT DISTINCT E1.cord, E1.ctext,
        ROW_NUMBER() OVER (PARTITION BY E1.id ORDER BY E1.id ASC, E1.cord ASC) AS entradaordem,
        COUNT(E1.id) OVER (PARTITION BY E1.id) AS tamanho_entrada
    FROM Entrada E1
    INNER JOIN conhecimento p2 ON E1.ctext = p2.ctext
    ORDER BY E1.id ASC
),
ConhecimentoRelevante AS (
    SELECT conh.cid, COUNT(DISTINCT conh.ctext) AS palavras_encontradas,
        COUNT(conh.ctext) OVER (PARTITION BY conh.cid) AS total_palavras_conhecimento,
        CAST(COUNT(DISTINCT conh.ctext) AS FLOAT) / CAST(MAX(E1.tamanho_entrada) AS FLOAT) AS porcentagem_relevancia
    FROM EntradaOrdenada E1
    INNER JOIN conhecimento conh ON E1.ctext = conh.ctext
    GROUP BY conh.cid
    ORDER BY palavras_encontradas DESC, porcentagem_relevancia DESC
    LIMIT 1
),
FraseCompleta AS (
    SELECT conh.cid, conh.cord, conh.ctext
    FROM conhecimento conh
    WHERE conh.cid = (SELECT cid FROM ConhecimentoRelevante)
    ORDER BY conh.cord
)
SELECT conh.cid, conh.cord, conh.ctext
FROM FraseCompleta conh
WHERE conh.cord <= (
    SELECT MIN(E1.cord) + (
        SELECT MAX(cord)
        FROM Entrada
    )
    FROM Entrada E1
)
ORDER BY conh.cord;



WITH Entrada AS (
    SELECT
        column1 AS id,
        column2 AS ctext,
        column3 AS cord
    FROM (VALUES
        (0,'do',1),
        (0,'Brasil',2),
        (0,'?',3)
    )  AS ptexts
),
EntradaOrdenada AS (
    SELECT DISTINCT E1.cord, E1.ctext,
        ROW_NUMBER() OVER (PARTITION BY E1.id ORDER BY E1.id ASC, E1.cord ASC) AS entradaordem,
        COUNT(E1.id) OVER (PARTITION BY E1.id) AS tamanho_entrada
    FROM Entrada E1
    INNER JOIN conhecimento p2 ON E1.ctext = p2.ctext
    ORDER BY E1.id ASC
),
ConhecimentoRelevante AS (
    SELECT conh.cid, COUNT(DISTINCT conh.ctext) AS palavras_encontradas,
        COUNT(conh.ctext) OVER (PARTITION BY conh.cid) AS total_palavras_conhecimento,
        CAST(COUNT(DISTINCT conh.ctext) AS FLOAT) / CAST(MAX(E1.tamanho_entrada) AS FLOAT) AS porcentagem_relevancia
    FROM EntradaOrdenada E1
    INNER JOIN conhecimento conh ON E1.ctext = conh.ctext
    GROUP BY conh.cid
    ORDER BY palavras_encontradas DESC, porcentagem_relevancia DESC
    LIMIT 1
),
FraseCompleta AS (
    SELECT conh.cid, conh.cord, conh.ctext
    FROM conhecimento conh
    WHERE conh.cid = (SELECT cid FROM ConhecimentoRelevante)
)
SELECT conh.cid, conh.cord, conh.ctext
FROM FraseCompleta conh
WHERE conh.cord <= (
    SELECT MAX(entradaordem)
    FROM EntradaOrdenada
) OR conh.ctext IN (SELECT ctext FROM EntradaOrdenada);



SELECT conh.cid, conh.cord, conh.ctext
FROM FraseCompleta conh
WHERE conh.cord <= (
    SELECT MAX(E1.cord)
    FROM EntradaOrdenada E1
)
ORDER BY conh.cord;
