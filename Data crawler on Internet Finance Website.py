from selenium import webdriver
driver = webdriver.Chrome(executable_path="C:\Python 3.6\chromedriver.exe")
print("driver kicks")
driver.get('http://shuju.wdzj.com/')
driver.find_element_by_xpath("//li[@data-type='30']").click()
print("find it")
driver.find_element_by_link_text('数据自选指标').click()
print("find it")

#js = 'document.getElementsByClassName("xlist condition-sel fl")[7].firstChild.nextSbiling.nextSbiling.nextSbiling.className="active");'
#js代码不能一句一句试运行 网页结构比较复杂 故采用模拟鼠标的行为点击不可点击的li元素
#driver.execute_script(js)
 
#点击 也就随时被js捕获 修改相应的点击元素的属性值

#driver.find_element_by_link_text('投资人数')
#print("find this unclickable li")

driver.find_elements_by_xpath("//li[@data-type='1']")[6].click()
driver.find_elements_by_xpath("//li[@data-type='2']")[4].click()


#就是找不见的节奏

#数据自选指标都能找到见
#原因是有多个这样子特征的元素
#需要更细致的定位
#垃圾网贷之家网站

#from selenium.webdriver import ActionChains
#ActionChains(driver).click(driver.find_element_by_link_text('投资人数')).perform()
#print("find it")



#li is not clickable

#driver.find_element_by_link_text('借款人数').click()
#print("find it")




driver.find_element_by_xpath("//button[@id='btn-diy']").click()

#driver.find_element_by_link_text('确定').click()
#print("find it")
#是用text就是找不见 未知错误 我也不知道为什么
#确定竟然也能找的见 因为这个元素的唯一的 返回一个object



#print(driver.find_elements_by_xpath("//td[@style='width:212.5px;']"))
#print(driver.find_elements_by_xpath("//td[@class='td-item']"))
#for each in driver.find_elements_by_xpath("//a[@target='_blank']"):
    #print(each.text)
#这样采集不回来js生成的数据
#有一种常用的可能是 js传回来数据更新DOM树 所以只要我们访问DOM树里的文本节点就好了
#还有一种可能是数据不是通过更新DOM树的 那就好玩了
#如何证明 这些数据是通过更新DOM树来更新的？？？ 

import csv
csvFile=open("C://Users//Dominic//Desktop//Finance_Data.csv","w+",newline='',encoding='utf-8')
writer=csv.writer(csvFile)
csvRow=['平台名称', '成交量（万元）','投资人数（人）', '借款人数（人）', '平均预期收益率（%）',
 
             '平均借款期限（月）', '待还余额（万元）'] #添加列属性名
        
writer.writerow(csvRow)

from bs4 import BeautifulSoup
html=driver.page_source

bsObj=BeautifulSoup(html,'lxml')

def find(ID):
    dataList=bsObj.find('tr',{'data-platid':ID}).findAll('td')
    print(dataList)
    data=[]
    data.append(bsObj.find('tr',{'data-platid':ID}).a.get_text())
    for i in dataList:
        data.append(i.div.get_text())
    print(data)
    adjust_data = []
    adjust_data.append(data[0])
    adjust_data.append(data[3])
    adjust_data.append(data[4])
    adjust_data.append(data[5])
    adjust_data.append(data[6])
    adjust_data.append(data[7])
    adjust_data.append(data[8])
    writer.writerow(adjust_data)

find('59')
find('1309')
find('85')
find('129')
find('142')
find('498')
find('268')
find('2016')
find('57')
find('689')
find('223') 


csvFile.close()
