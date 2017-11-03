import convert_data_to_tfrecords
import import_data


flo_data = import_data.read_flo_file('flow10.flo')

converter = convert_data_to_tfrecords.TFRecordsConverter('SAME','./')
converter.convert_file('frame10.png','frame11.png',flo_data)
