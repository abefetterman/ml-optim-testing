import os
from sacred.observers import MongoObserver
from sacred.observers import FileStorageObserver

def add_observer(ex):
    mongourl = os.environ.get('SACRED_MONGOURL')
    dbname = os.environ.get('SACRED_DBNAME', 'sacred')
    if (mongourl != None):
        ex.observers.append(MongoObserver.create(url=mongourl, db_name=dbname))
    else:
        ex.observers.append(FileStorageObserver.create('logs'))
