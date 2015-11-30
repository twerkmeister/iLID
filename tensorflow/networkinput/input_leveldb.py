import caffe
import leveldb
import numpy as np
from caffe.proto import caffe_pb2

DB_PATH = "/Users/therold/Downloads/dubsmash/db"

db = leveldb.LevelDB(DB_PATH)
datum = caffe_pb2.Datum()

for key, value in db.RangeIter():
    datum.ParseFromString(value)

    label = datum.label
    data = caffe.io.datum_to_array(datum)