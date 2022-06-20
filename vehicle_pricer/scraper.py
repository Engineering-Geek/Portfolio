from bs4 import BeautifulSoup
import requests
from typing import Dict, List
from tqdm import tqdm
import json
import pandas as pd
import os


"""NOT FINISHED YET"""


class Vehicle:
    def __init__(self, vin: str, name: str, price: str, url: str, image_links: List[str], description: str):
        self.vin = vin
        self.name = name
        self.price = price
        self.url = url
        self.image_links = image_links
        self.description = description


class Vehicles():
    def __init__(self):
        self.vehicles = []
    
    def append(self, vehicle: Vehicle):
        self.vehicles.append(vehicle)
    
    def pop(self, index: int):
        return self.vehicles.pop(index)
    
    def __len__(self):
        return len(self.vehicles)
    
    def __getitem__(self, index: int):
        return self.vehicles[index]
    
    def pandas(self):
        df = pd.DataFrame(self.vehicles)
        df.set_index('vin', inplace=True)
        return df
    
    def save(self, dirpath: str):
        df = self.pandas()
        df.to_csv(os.path.join(dirpath, 'vehicles_{}.csv'.format(len(os.listdir(filepath)))))


class AutotraderScraper:
    def __init__(self, province: str, city: str):
        self.soup = None
        self.results_per_page = 50
        self.province = province
        self.city = city
        self.url = "https://www.autotrader.ca"
        self.params = {
            "rcp": 50,
            "rcs": 0,
            "srt": 50,
            "prx": 50,
            "prv": self.province.lower(),
            "hprc": True,
            "wcp": True,
            "inMarket": True,
        }
        self.header = {'User-Agent': 'Mozilla/5.0'}
    
    def _get_province_abbreviation(self, province: str) -> str:
        """
        Returns the province abbreviation for the given province name.
        """
        province_abbreviations = {
            "Alberta": "ab",
            "British Columbia": "bc",
            "Manitoba": "mb",
            "New Brunswick": "nb",
            "Newfoundland and Labrador": "nl",
            "Nova Scotia": "ns",
            "Ontario": "on",
            "Prince Edward Island": "pe",
            "Quebec": "qc",
            "Saskatchewan": "sk",
            "Northwest Territories": "nt",
            "Nunavut": "nu",
            "Yukon": "yt"
        }
        return province_abbreviations[province]
    
    def _get_page_links(self, total_results: int=-1, results_per_page: int=50) -> List[str]:
        url = self.url + "/cars/{}/{}/".format(self._get_province_abbreviation(self.province), self.city)
        page_links = []
        
        # getting first page to set up a few things
        request = requests.get(url, params=self.params, headers=self.header)
        soup = BeautifulSoup(request.text, "html.parser")
        print(request.url)
        
        # setting up the maximum number of listings we can have, and setting up the number of listings per page
        max_listings = int(soup.find("span", {"id": "sbCount"}).text.replace(",", ""))
        total_results = max_listings if max_listings > results_per_page or max_listings == -1 else total_results
        
        for index in tqdm(range(1, int(total_results / results_per_page) + 1), desc="Getting page links"):
            self.params["rcs"] += results_per_page
            soup = BeautifulSoup(requests.get(url, params=self.params, headers=self.header).text, "html.parser")
            search_listings = soup.find("div", {"id": "SearchListings"})
            
            for listing in search_listings.find_all("div", {"class": "result-item"}):
                main_picture = listing.find("a", {"class": "main-photo"})
                page_link = main_picture.get("href")
                page_links.append(page_link)
            if index == 1:
                break
        return list(set(page_links))
    
    def _regex_hell(self, script: str) -> str:
        """
        takes in the script element and converts it into json
        """

    def _extract_vehicle_info(self, page_link: str) -> Vehicle:
        url = self.url + page_link
        request = requests.get(url, headers=self.header)
        raw_text = request.text
        # After inspecting the html, I noticed that the data for the whole website is in this script element,
        #   specifically the script element that contains the json data. I'm going to manually extract it like this
        #   and you can mail your complaints about unreadability to idgaf@shove.it.up.com. I did what I could.
        start_index = raw_text.find("window['ngVdpModel'] = ") + len("window['ngVdpModel'] = ")
        raw_text = raw_text[start_index:]
        end_index = raw_text.find("window['ngVdpGtm'] = ")
        raw_text = raw_text[:end_index].strip()[:-1]
        data = json.loads(raw_text)
        
        print(data)
        exit()


if __name__ == "__main__":
    scraper = AutotraderScraper("Ontario", "waterloo")
    links = scraper._get_page_links(100)
    # find duplicates in the links
    links = list(set(links))
    scraper._extract_vehicle_info(links[0])
    # print(requests.get("https://www.autotrader.ca/cars/on/waterloo/?rcp=50&rcs=0&srt=50&prx=50&prv=ontario&hprc=True&wcp=True&inMarket=True"))
