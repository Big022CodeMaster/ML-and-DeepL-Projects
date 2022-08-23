import requests

#https://proxywbs:3128
proxies = {
    "http": "http://proxywbs:3128" ,
    "https": "https://proxywbs:3128"
}

url = "http://httpbin.org/ip"

r = requests.get(url, proxies=proxies)

print(r.json())