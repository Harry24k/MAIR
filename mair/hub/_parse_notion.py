import json
import requests

NOTION_API_TOKEN = 'secret_Adp7AI8ugJpicvcKQDAbjpuN7Ah6fbgJ6JnVilOHzFj'
NOTION_DATABASE_ID = "958ba2d81d194c1fa86accf65c1f6b9e"
NOTION_DATABASE_URL = f'https://api.notion.com/v1/databases/{NOTION_DATABASE_ID}/query'


def get_text_from_res(res, key):
    return res[key]['title'][0]['plain_text']


def get_single_from_res(res, key):
    return res[key]['select']['name']

def get_multi_from_res(res, key):
    items = []
    for item in res[key]['multi_select']:
        items.append(item['name'])
    return items


def get_value_from_res(res, key):
    return res[key]['number']


def get_checkbox_from_res(res, key):
    return res[key]['checkbox']


def get_link_from_res(res, key):
    urls = {}
    for item in res[key]['files']:
        name = item['name']
        assert name.replace(".pth", "") == key.lower()
        url = item['file']['url']
        urls[name] = url
    return urls[name]


def print_format(name, key):
    print("{0:<15}: ".format(name) + str(key))

def get_link_by_id(id, flag):
    headers = {'Authorization': f'Bearer {NOTION_API_TOKEN}',
               'Notion-Version': '2022-02-22',
               'Content-Type': 'application/json'}
    payload = {'page_size': 100}
    has_more = True
    while has_more is True:
        res = requests.post(NOTION_DATABASE_URL, data=json.dumps(payload), headers=headers).json()
        has_more = res['has_more']
        payload = {'start_cursor':res['next_cursor']}

        for r in res['results']:
            prop = r['properties']
            if len(prop["ID"]['title']) == 0:
                continue
            if get_text_from_res(prop, "ID") != id:
                continue
            print("="*60)
            print(id)
            print("="*60)
            
            print_format("Method", get_single_from_res(prop, "Method"))
            print_format("Architecture", get_single_from_res(prop, "Architecture"))
            print_format("Batch Size", get_single_from_res(prop, "Batch Size"))
            print_format("Aug.", get_checkbox_from_res(prop, "Aug."))
            print_format("AWP", get_checkbox_from_res(prop, "AWP"))
            print_format("Extra Data", get_checkbox_from_res(prop, "Extra Data"))
            for measure in ["Clean(Last)", "PGD(Last)", "Clean(Best)", "PGD(Best)"]:
                value = get_value_from_res(prop, measure)
                if value:
                    print_format(measure, "%2.2f%%"%value)
#             print("Detail Info: "+r['url'])
            print("="*60)
            return get_link_from_res(prop, flag)
