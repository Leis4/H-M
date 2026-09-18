--IMPORT DATA (Используем относительные пути)--
CREATE TABLE IF NOT EXISTS articles AS
SELECT * FROM read_csv_auto('data/archive/articles.csv');

CREATE TABLE IF NOT EXISTS transactions AS
SELECT * FROM read_csv_auto('data/archive/transactions_train.csv');

CREATE TABLE IF NOT EXISTS customers AS
SELECT * FROM read_csv_auto('data/archive/customers.csv');
------------------------------------------

--STRUCTURE--
SELECT * FROM articles LIMIT 5;
SELECT * FROM transactions LIMIT 5;
SELECT * FROM customers LIMIT 5;
------------------------------------------

--WORKING WITH DATA--
CREATE OR REPLACE TEMP VIEW articles_prepare AS (
    SELECT
        article_id,
        prod_name,
        product_type_name,
        product_group_name,
        graphical_appearance_name,
        colour_group_name,
        perceived_colour_value_name,
        perceived_colour_master_name,
        department_name,
        index_name,
        index_group_name,
        section_name,
        garment_group_name,
        COALESCE(detail_desc, '') AS clean_desc,
        LENGTH(COALESCE(detail_desc, '')) AS desc_length
    FROM articles
);

CREATE OR REPLACE TEMP VIEW transactions_with_customers AS (
    SELECT
        t.article_id,
        t.price,
        c.age
    FROM transactions t
    LEFT JOIN customers c ON t.customer_id = c.customer_id
);

CREATE OR REPLACE TEMP VIEW transactions_prepare AS (
    SELECT
        article_id,
        AVG(price) AS target_price,     --TARGET FEATURE
        COUNT(*) AS total_count,
        AVG(age) AS avg_buyer_age
    FROM transactions_with_customers
    GROUP BY article_id
    HAVING COUNT(*) >= 5
);
------------------------------------------

-- UNION ALL TABLES--
CREATE TABLE IF NOT EXISTS final_features AS
SELECT
    tr.article_id,
    tr.target_price,
    tr.total_count,
    tr.avg_buyer_age,

    art.prod_name,
    art.product_type_name,
    art.product_group_name,
    art.graphical_appearance_name,
    art.colour_group_name,
    art.perceived_colour_value_name,
    art.perceived_colour_master_name,
    art.department_name,
    art.index_name,
    art.index_group_name,
    art.section_name,
    art.garment_group_name,
    art.clean_desc,
    art.desc_length
FROM transactions_prepare tr
LEFT JOIN articles_prepare art ON tr.article_id = art.article_id;
------------------------------------------

--CLEAN--
DROP TABLE IF EXISTS articles;
DROP TABLE IF EXISTS transactions;
DROP TABLE IF EXISTS customers;
------------------------------------------

--TEST--
SELECT * FROM final_features LIMIT 5;
------------------------------------------