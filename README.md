# model experimentation

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

A short description of the project.

## 1. Executive Summary

The project aims to develop a machine learning data product for predicting airfare prices for local cities, leveraging various factors such as flight characteristics and historical pricing data. Each team member independently trains models, with the goal of providing users accurate airfare on domestic flights.
The report details each team member’s methodology, including data preparation, feature engineering, model selection, and result evaluation. By integrating individual strategies and their outcomes, the team has created robust machine learning models for predicting airfare prices. These models are deployed as a web service, offering users reliable and up-to-date airfare predictions

## 2. Business Understanding

a. Business Use Cases
Frequent Flyers Program
A successful model will enable airlines to identify the most attractive frequent flyer programs and offer tailored loyalty deals to customers based on seasonality and network demand. This approach enhances customer retention in a competitive domestic airline market, allowing for lower-cost flight operations and optimal resource allocation of planes and personnel.
Travel Agency Optimisation
Predictions from the model will improve travel agencies’ ability to assess the likelihood of price fluctuations. This enhances the service offerings for travel agents by allowing them to identify suitable, tailored travel deals for their clients across multiple airline options.
Airline Pricing Strategy
Airlines can use the models to optimise their pricing strategies and marketing campaigns by assessing the demand characteristics for flights between destinations. This informed approach helps airlines set competitive prices, which is crucial for maximising flight occupancy, boosting revenue, and enhancing customer satisfaction. Additionally, the service allows airlines to identify domestic travel networks, enabling better resource utilisation for higher-demand routes.

## 3. Data Understanding
The dataset provided for this project was acquired from Expedia, an aggregator of travel information. It contains 13519999 flight records and 23 features. Figure 3.1 shows the composition of the features and their data types.

![unnamed](https://github.com/user-attachments/assets/2ae90264-bb49-47e3-a364-59ed46341077)

The total fare amount for flight records ranges from $23 to $8200. The majority of the flights are priced somewhere between $300-$500.

<img width="633" alt="Screenshot 2024-12-13 at 11 53 32 PM" src="https://github.com/user-attachments/assets/34812e42-2085-4303-a79c-317153677a41" />

The dataset contains flight itineraries for the year 2022 and months ranging from April to July. The highest fare amount rate was recorded during the month of April and the lowest was during May/June.

![unnamed (1)](https://github.com/user-attachments/assets/c1e5cfdc-a765-4a3a-aa69-32aeebd08c91)

There are 16 unique airport locations included as the origin and destination. These values are entered in the form of IATA codes for each location. Figure 3.4 shows the IATA code and the locations it represents.

![unnamed (2)](https://github.com/user-attachments/assets/e5070554-b8d2-4b87-82c5-5cde185398b2)

LAX has the highest number of departures, followed by LGA, indicating that these airports experience significant travel activity during these months. Subsequently, the most expensive origin-destination pair is LGA->OAK, followed by IAD->OAK and the cheapest is BOS->LGA.

![unnamed (3)](https://github.com/user-attachments/assets/f646ab08-cb3e-4fc4-a104-62dc91138691)

![unnamed (4)](https://github.com/user-attachments/assets/8b3cdad9-4b2b-4855-a823-fb85dbdb2c06)

The travel duration and distance show a linear relation with the total fare. Usually, if the distance or duration increases the flight prices increase. This is due to the increases in resources required for a flight to cover larger distances.

![unnamed (5)](https://github.com/user-attachments/assets/9af1403f-88a2-45eb-8139-86421cec495c)

![unnamed (6)](https://github.com/user-attachments/assets/c58abed6-ba62-477f-ae4e-6c7a7df1c77f)

The majority of the people travel in basic economy, this could be due to the higher fares related to
premium seating options. The average price for non-basic seats is almost double than that of basic.

![unnamed (7)](https://github.com/user-attachments/assets/8c2b16a7-dece-4e8c-86d7-97de8152e595)

![unnamed (8)](https://github.com/user-attachments/assets/1607bffb-5708-4c88-92a2-545d38e80b38)

The majority of flights include stops, often chosen for two main reasons. Firstly, travelers may prefer to visit additional locations beyond their primary destination. Secondly, connecting flights are typically more cost-effective than non-stop flights. However, based on the graph shown below, it can be noted that pricing for non stop flights is cheaper.

![unnamed (9)](https://github.com/user-attachments/assets/4e2cdb23-e5eb-4bb7-837d-0c761d9ee46c)

![unnamed (10)](https://github.com/user-attachments/assets/22cadde0-b807-4abe-b7b1-a49f0e7ac799)

## 4. Data Preparation
The raw dataset was a combination of multiple folders. Each folder contains itineraries related to its specific airport code. These files were in zip format. A function was created to unzip all these files and create a single dataframe/csv.

![unnamed (11)](https://github.com/user-attachments/assets/87f4b35e-d768-4196-b017-40ef92aff9be)

The travel duration was converted from ISO 8601 format to minutes using a Python function. For example, PT6H15M is a representation of a duration in ISO 8601 format. This specific string indicates a period lasting 6 hours and 15 minutes.

![unnamed (12)](https://github.com/user-attachments/assets/5cb05d7d-35fd-4921-a99e-bb0a955727fc)

Columns such as cabin type, departure time epoch seconds, departure time raw, etc contain multiple segment data. This is due to the concept of connecting flights where each value corresponds to its specific segment. They contain “||” symbol between them which helps identify the segments. These were split into columns for each segment.

![unnamed (13)](https://github.com/user-attachments/assets/8b1b5bfe-1781-4734-8c9c-90b8ebb480c4)

The missing values were summarised, and the segmented parts had the majority of values. This is because there are very limited flights with 2 or 3 stops. In this case, the variable “No cabin” was used to replace missing values. In the case of totalTravelDistance, the missing values were replaced with the median distance for the specific origin-destination pair.

![unnamed (14)](https://github.com/user-attachments/assets/23b66e64-e5a3-444c-854e-237c4be54379)

After replacing missing values, a new feature called num_stop was created to track the number of stops for each flight record. The first segment of the departure_date was then extracted and converted to a datetime format with timezone awareness, assuming it was originally in the UTC time zone. This date was subsequently converted to the Australian time zone, as the app users are located in Australia. Finally, the date was split into individual components: day, month, year, hour, and minute.

Using the departure datetime, by subtracting it from the search date the gap between these two values was generated and added to a column “days_difference”

![unnamed (15)](https://github.com/user-attachments/assets/07c0675d-c245-437e-822d-505d004181a6)

![unnamed (16)](https://github.com/user-attachments/assets/6b9e0bc3-5573-46ed-88a9-27adec7dcf6d)

The final step for data preparation is to remove unwanted columns. The columns that were not dropped were specifically chosen as they were a part of the inputs that will be taken from the users, also, some columns were left out which will be filled with derived information based on user’s input.

![unnamed (17)](https://github.com/user-attachments/assets/25eb3a02-e9c0-4885-a6cd-4d130994cc33)

![unnamed (18)](https://github.com/user-attachments/assets/0ea55c9b-6540-4fe3-b4a8-a85a06636142)



## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         model_experimentation and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── model_experimentation   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes model_experimentation a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

--------

