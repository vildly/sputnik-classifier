
// init.js
// Create a sample collection and insert some documents
db = db.getSiblingDB("testdb");

db.createCollection("sample");
db.sample.insertMany([
  { name: "Document 1", createdAt: new Date() },
  { name: "Document 2", createdAt: new Date() }
]);
