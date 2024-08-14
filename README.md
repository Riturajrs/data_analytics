
# Store Monitoring Django App

## Overview

This Django application monitors the uptime and downtime of various stores. It provides reports on store availability, handling timezones and business hours for accurate metrics. The application is designed to be scalable and efficient, leveraging background tasks for data processing.

## Features

* **Store Monitoring:** Tracks store uptime and downtime.
* **Timezone Handling:** Adjusts times based on different store timezones.
* **Periodic Polling:** Collects data at regular intervals.
* **Business Hours Management:** Handles store-specific business hours for accurate reporting.
* **Report Generation:** Provides detailed reports on store availability.

## Extrapolation of data points:

A Naive Bayes model is being used for predicting the status of the store if the observation for that particular timestamp is not available. Each store has it's own model, which is being first trained on the observation data, and later being used to extrapolate in order to calculate uptime and downtime.

For training the data, we are storing the states from the observations along with the minutes which have passed since the business's start time. Now, we can predict the status of any minute from the business's start time to get the status of the store.

## Uptime and downtime calculation:

### Last week:

For this we are iterating through all the weekdays present in the observation. The assumption here is that we are only storing the observations for last 7 days, as observations earlier than that are not required. For each weekday we are going through the business hours of that day. We are starting at start time of the business and every time we are hopping one hour until the business end time is reached. If there is no observation for a particular timestamp, for which are polling the status, we are using the machine learning model which we trained earlier to predict the status.

### Last day:

For this, we are first finding out the latest weekday, after we have got that we are going through the business hours of that day. We are starting at start time of the business and every time we are hopping one hour until the business end time is reached. If there is no observation for a particular timestamp, for which are polling the status, we are using the machine learning model which we trained earlier to predict the status.

### Last hour:

For calculating uptime and downtime at last hour, we are first searching for the end time of the last day, for which we have observations. Once we have the end time, we are going back 1 hour, and are polling for the status every mintue until we have reached the end time of the business. If we do not have any observation for a particular mintue, we are using the machine learning model for predicitng the status of that minute.
