-- init.sql

CREATE TABLE IF NOT EXISTS devices (
    id SERIAL PRIMARY KEY,  -- 'SERIAL' makes 'id' auto-increment
    name VARCHAR(100) NOT NULL,
    version VARCHAR(50) NOT NULL
);
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(255) NOT NULL UNIQUE,
    password_hash CHAR(64) NOT NULL
);

CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- If you manually want to assign an ID, it's possible even with SERIAL
-- By specifying 'id', you override the default serial behavior, just for the initial seed.
-- INSERT INTO devices (id, name, version) VALUES
-- (1, 'Device A', '1.0'),
-- (2, 'Device B', '1.1'),
-- (3, 'Device C', '2.0');

INSERT INTO users (username, password_hash)
VALUES ('your_username', ENCODE(DIGEST('your_password', 'sha256'), 'hex'));
