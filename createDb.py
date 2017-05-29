import tinydb

def create_db(dbName):
    db = tinydb(dbName)
    return(db)