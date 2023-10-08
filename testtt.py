import time
import requests
import re
import json
import random
time_t=str(int(time.time()))
url='https://cpipc.acge.org.cn/login/doCheckId4DocumentRetrievePwd'

headers = {
'Accept': '*/*',
'Accept-Encoding':'gzip, deflate, br',
'Accept-Language':'zh-CN,zh;q=0.9',
'Connection':'keep-alive',
'Content-Length':'291',
'Content-Type': 'application/x-www-form-urlencoded; charset=UTF-8',
'Cookie': 'Hm_lvt_efff57047d75583f6c463eaee32793c4=1695284855,1695341546,1695685706,1696139358; JSESSIONID=27189D99B0222F489C7D66D8AC71E6CB; Hm_lpvt_efff57047d75583f6c463eaee32793c4=1696139572; SERVERID=1cda3474da039dd5b95149633f9c09f6|'+time_t+'|1696139350',
'Host':'cpipc.acge.org.cn',
'Origin':'https://cpipc.acge.org.cn',
'Referer':'https://cpipc.acge.org.cn/login/retrievePwd?retrievePwdType=3',
'Sec-Ch-Ua': '"Chromium";v="116", "Not)A;Brand";v="24", "Google Chrome";v="116"',
'Sec-Ch-Ua-Mobile' : '?0',
'Sec-Ch-Ua-Platform' : '"Windows"',
'Sec-Fetch-Dest':'empty',
'Sec-Fetch-Mode':'cors',
'Sec-Fetch-Site':'same-origin',
'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36',
'X-Requested-With':'XMLHttpRequest'

}
# print(headers)

# idd=1
# heart_data = {"userName":"裴庆祺",
#      "organName":"西安电子科技大学",
#      "mobile":"155"+"{:0>4}".format(idd)+"1381"
#      # "email": null
#      }
# heart_data.append(
#     {"userName":"裴庆祺",
#      "organName":"西安电子科技大学",
#      "mobile":"155"+"{:0>4}".format(idd)+"1381",
#      "email":"null"
#      }
#
# )

data=("pcpServiceVariableOb=%7B%7D&identityInfo=%7B%22userName%22%3A%22%E8%A3%B4%E5%BA%86%E7%A5%BA%22%2C%22organName%22%3A%22%E8%A5%BF%E5%AE%89%E7%94%B5%E5%AD%90%E7%A7%91%E6%8A%80%E5%A4%A7%E5%AD%A6%22%2C%22mobile%22%3A%2215500091381%22%2C%22email%22%3Anull%7D&token=27189D99B0222F489C7D66D8AC71E6CB")
r = requests.post(url=url,headers=headers,json=data)
print(r.text)
# print(data)