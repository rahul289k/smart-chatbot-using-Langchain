import json

file_path = '/Users/rahulkumar/Downloads/json_temp/datahub_mysql_sample.json'
dump_file_path = '/Users/rahulkumar/Downloads/clean_datahub_mysql_900.json'

with open(file_path, 'r') as json_file:
    data = json.load(json_file)
data_to_dump = []
for each_json_obj in data:
    data_dict = {}
    for each_tag in each_json_obj:
        if each_tag == 'urn':
            data_dict['major_tag'] = each_json_obj[each_tag].split(':')[2]
        else:
            data_dict[each_tag] = each_json_obj[each_tag]
    data_to_dump.append(data_dict)

with open(dump_file_path, 'w') as dump_file:
    json.dump(data_to_dump, dump_file)
