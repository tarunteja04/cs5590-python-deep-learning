import urllib.request
from bs4 import BeautifulSoup


url = "https://en.wikipedia.org/wiki/List_of_states_and_territories_of_the_United_States"
fpURL= urllib.request.urlopen(url)

#parsing the givenhtml page
soup = BeautifulSoup(fpURL, "html.parser")

#finfing table im the page

table = soup.find("table")

#finding table headers from the page
table_header = table.find_all('th')

#iterating through the headers and displaying the headers in text format
th = [t.text for t in table_header]
print(th)

#saving the output to a file
file = open("output.txt", "w")

#writing the table headers to the file
file.write(str(th))

#finding the table rows in the tabel
tables_row = table.find_all("tr")

for tr in tables_row:   #for each row in the table
        td = tr.find_all('td')   #finfing all the table cells in the table, td stands for table cells
        row = [r.text for r in td]   #displaying each row and cell in text format
        file.write("\n") #wrrting each row in new line
        print(row)
        file.write(str(row)) #writing the output

