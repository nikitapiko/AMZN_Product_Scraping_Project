import xml.etree.ElementTree as ET
from lxml import html
import pandas as pd
import json

# Parse the XML file
tree = ET.parse('output.xml')
root = tree.getroot()

product_dict = {}
idx = 0
topic = ''

prod_counter = 0
# Extract and print the text content of <msg> tags
for i, msg in enumerate(root.findall(".//msg")):
    html_code = msg.text

    if "a-price-whole" in html_code \
            and \
            "a-size-base s-underline-text" in html_code \
            and \
            "a-icon-alt" in html_code:
        print("FIRST IFS - CHECK")
        prod_counter += 1
        # print(prod_counter)
        # print(html_code)
        tree = html.fromstring(html_code)

        # Use XPath to locate the element with the specified class
        if "a-size-medium a-color-base a-text-normal" in html_code:
            title = tree.xpath('//*[@class="a-size-medium a-color-base a-text-normal"]')[0].text_content()
        elif "a-size-mini a-spacing-none a-color-base s-line-clamp-4" in html_code:
            title = tree.xpath('//*[@class="a-size-mini a-spacing-none a-color-base s-line-clamp-4"]')[0].text_content()
        elif "a-size-base-plus a-color-base a-text-normal" in html_code:
            try:
                title = tree.xpath('//*[@class="a-size-base-plus a-color-base a-text-normal"]')[0].text_content()
            except:
                title = 'None'
        # price = tree.xpath('//span[@class="a-price-whole"]')[0].text_content()
        price = tree.xpath('//span[@class="a-offscreen"]')[0].text_content()   #<span class="a-price-whole">59,48</span> #a-offscreen
        rating_count = tree.xpath('//span[@class="a-size-base s-underline-text"]')[0].text_content()
        rating_stars = tree.xpath('//span[@class="a-icon-alt"]')[0].text_content()

        print(title)
        print(price)
        print(rating_count)
        print(rating_stars)
        print("")


        idx += 1

        product_dict[f"product_{idx}"] = {
                'title': title,
                'price_eur': float(price[:-2].replace('.', '').replace(",", ".")),
                'review_count': int(rating_count.replace(".", "")),
                'rating_stars': float(rating_stars[:3].replace(",", "."))
            }

df = pd.DataFrame(product_dict).T
df = df.reset_index()
df = df.rename(columns={'index': 'product_id'})

# ______ SAVE TABLE _____
df.to_csv('product_table.csv', index=False)






first_50_products = df.sort_values(by='review_count', ascending=False)[:50].to_string()

# print(first_50_products)



with open('product_table.csv') as f:
    first50 = ''.join(f.readlines()[:51])
    # for line in range(51):
    #     print(f.readline())

print(first50)


        # # Extract the text content of the element
        # if title and price:
        #     title_text = title[0].text_content()
        #     price_text = price[0].text_content()
        #     print(title_text)
        #     print(price_text)
        # else:
        #     print("Element not found")
#         idx += 1
#         # Extract and print the text after "Text inside the element:"
#         text = text.split("Text inside the element:")[1].strip()
#         # print(text)
#         # print('')
#
#         product_dict[f"product_{idx}"] = text
#
# BLACKLIST = ['Vorgestellte Produkte von Amazon-Marken', 'Amazons Tipp', 'Gesponsert', 'Befristetes Angebot', 'Bestseller']
#
# for product in product_dict.keys():
#     product_items = [item for item in product_dict[product].split('\n') if item not in BLACKLIST][:4]
#     print(product_items)
#     print("")
#     title = product_items[0]
#     price = product_items[2] + ',' + product_items[3]
#     print('Your Alleged Price: ' + price)
#     print("\n\n")
#     review_count = product_items[1]
#     # missing the rating star count -> next step is to search by element for sure -> advanced
#
#     product_dict[product] = {
#         'title': title,
#         'price': price,
#         'review_count': review_count
#     }


