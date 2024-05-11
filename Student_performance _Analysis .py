#!/usr/bin/env python
# coding: utf-8

# # Analysis of Student Performance Data for Anomaly Detection and Pattern Recognition  
#                                                                                          -Lavanya Ravilla 
# 
# Background: 
# 
# In this analysis, you will explore a dataset representing the daily performance of 10 students over a 30-day period.
# 
# 
# Dataset Overview: Each student attempted between 35 to 50 questions daily. These questions were distributed over four chapters, with varying levels of difficulty and engagement requirements. The performance status for each question is recorded as "Correct", "Incorrect", or "Unattempted".
# 
# 
# Objectives:
# 
# 
# Trend Analysis: 
# 
# Identify and visualize the trends in student performance over the 30-day period. Analyze these trends by chapter and across the entire dataset.
# 
# Anomaly Detection:
# 
# Fatigue or Boredom Effects: Detect any days with an unusual increase in "Incorrect" and "Unattempted" answers, which might suggest fatigue or boredom, particularly in the middle of longer chapters.
# 
# Impact of External Events: Identify days where there is a noticeable spike in "Unattempted" answers, potentially indicating external events or disruptions affecting student engagement.
# 
# Random Performance Fluctuations: Spot patterns where student answers reflect randomness, suggesting days when students struggled significantly with the material.
# 
# Pattern Recognition:
# 
# Determine if there is a learning curve visible where students start with poorer performance at the beginning of each chapter but improve as they become more familiar with the content.
# 
# Assess whether performance varies significantly from the start to the end of each chapter, suggesting review or fatigue effects.
# 
# Recommendations: 
# 
# Based on your analysis, provide recommendations on potential academic interventions that could help improve student engagement and performance. 
# 
# Deliverables:
# 
# A comprehensive report including graphs and tables to support your analysis.

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


import warnings

warnings.simplefilter("ignore", UserWarning)


# In[3]:


df=pd.read_excel("Student_Performance_Dataset_3.xlsx")


# In[4]:


df.shape


# In[5]:


df.info()


# In[6]:


df.head()


# In[7]:


df.isna().sum()


# In[8]:


df.columns


# In[9]:


df["Status"].unique()


# Anomaly Detection:

# 1.Fatigue or Boredom Effects: Detect any days with an unusual increase in "Incorrect" and "Unattempted" answers, which might suggest fatigue or boredom, particularly in the middle of longer chapters.

# In[10]:


def analyze_fatigue_boredom(data):
    daily_incorrect_unattempted = (
          data[data["Status"].isin(["Incorrect", "Unattempted"])]
          .groupby(["Day", "Chapter"])
          .size()
          .to_frame(name="Count")
          .reset_index()
      )
    
    total_questions = (
      data.groupby(["Day", "Chapter"])["Question ID"]
      .size()
      .to_frame(name="Total Questions")
      .reset_index()
    )
    
    daily_incorrect_unattempted = daily_incorrect_unattempted.merge(total_questions, on=["Day", "Chapter"])
    
    daily_incorrect_unattempted["Percentage"] = (
      daily_incorrect_unattempted["Count"] / daily_incorrect_unattempted["Total Questions"]) * 100
    
    print("before rolling daily_incorrect_unattempted shape",daily_incorrect_unattempted.shape)
    
    rolling_means=daily_incorrect_unattempted.groupby("Chapter")["Percentage"].rolling(window=5).mean().reset_index()
    
    daily_incorrect_unattempted = daily_incorrect_unattempted.merge(rolling_means, on="Chapter")
    daily_incorrect_unattempted.rename(columns={"Percentage_y": "Rolling Mean"}, inplace=True)
    print("daily_incorrect_unattempted with rolling",daily_incorrect_unattempted.head(2))
    
    daily_incorrect_unattempted[daily_incorrect_unattempted["Rolling Mean"].isna()].head(10)
    
    daily_incorrect_unattempted.drop_duplicates()
    
    daily_incorrect_unattempted["Rolling Mean"].fillna(method='bfill', inplace=True)   
    daily_incorrect_unattempted.head(10)
    
    daily_incorrect_unattempted=daily_incorrect_unattempted.drop(columns=['level_1'],axis=1)
    
    fatigue_plots = {}
    days_with_potential_fatigue = []
    def identify_fatigue_days_iqr(data):
        for i in daily_incorrect_unattempted["Chapter"].unique():
            filtered_data = daily_incorrect_unattempted[daily_incorrect_unattempted["Chapter"] == i]        

            Q1 = filtered_data['Percentage_x'].quantile(0.25)
            Q3 = filtered_data['Percentage_x'].quantile(0.75)
            IQR = Q3 - Q1


            outliers = filtered_data[(filtered_data['Percentage_x'] < (Q1 - 1.5 * IQR)) | (filtered_data['Percentage_x'] > (Q3 + 1.5 * IQR))]
            days_with_potential_fatigue.extend(outliers["Day"].tolist())
        return days_with_potential_fatigue

    days_with_fatigue_iqr = identify_fatigue_days_iqr(daily_incorrect_unattempted.copy())
    print(days_with_fatigue_iqr)
    
    
    fatigue_plots = {}
    window=5
    for i in daily_incorrect_unattempted["Chapter"].unique():
    
        filtered_data = daily_incorrect_unattempted[daily_incorrect_unattempted["Chapter"] == i]
    fig, ax = plt.subplots()

    ax.plot(
        filtered_data["Day"], filtered_data["Percentage_x"], label="Daily Percentage"
    )
    ax.plot(
        filtered_data["Day"], filtered_data["Rolling Mean"], label="Rolling Mean (Window: {})".format(window)
    )
    
    ax.set_title(f"Incorrect/Unattempted Rate")
    ax.set_xlabel("Day")
    ax.set_ylabel("Percentage")
    ax.legend()
    fatigue_plots[i] = fig

    return {
            "days_with_potential_fatigue": days_with_potential_fatigue,
          "fatigue_plots": fatigue_plots,
      }


# In[11]:


analyze_fatigue_boredom(df)


# inference:
# Potential Spikes in Unattempted Answers:
# 
# There are several days where the line spikes, indicating a significant increase in the percentage of incorrect/unattempted responses. These days are potentially: Day 26, Day 28, Day 30, and possibly Day 27.
# Possible Reasons for Spikes:
# 
# School holidays or breaks
# School assemblies or field trips
# Substitute teachers
# Technological issues
# Unexpected events in the classroom
# 
# Days with Lower Unattempted Rates:
# 
# Conversely, there are days with a lower percentage of unattempted responses (e.g., Day 25, Day 27, Day 28).indicate Easier topics or assessments
# Increased student engagement or focus
# More effective teaching method
# 
# Day 21 is considered as Fatigue chapter3  

# In[ ]:





# 2.Impact of External Events: Identify days where there is a noticeable spike in "Unattempted" answers, potentially indicating external events or disruptions affecting student engagement.

# In[12]:


def identify_spike_days(df, threshold=0.2):
    untempted_counts =df[df["Status"]=="Unattempted"].groupby([ "Chapter","Day"]).size().reset_index()
    untempted_counts_df = pd.DataFrame(untempted_counts)
    untempted_counts_df.columns = ["Chapter", "Day", "Count"]
    
    daily_total=df.groupby([ "Chapter","Day"]).size().reset_index()
    daily_total_df=pd.DataFrame(daily_total)
    daily_total_df.columns=["Chapter", "Day", "Count1"]
    
    
    # protportion on unattempted in chapter and day
    unattermp_daily_count=untempted_counts_df.merge(daily_total_df,on=['Chapter','Day'])
    unattermp_daily_count['Prop']=unattermp_daily_count['Count']/unattermp_daily_count['Count1']
    

    spike_days = unattermp_daily_count[unattermp_daily_count['Prop'] > threshold].index.tolist()
    
    
    plt.figure(figsize=(10, 6))
    plt.plot(unattermp_daily_count['Day'], unattermp_daily_count['Prop'])
    plt.xlabel('Day')
    plt.ylabel('Unattempted Answer Rate')
    plt.title('Daily Unattempted Answer Rate')
    plt.grid(True)

    
    
    plt.figure(figsize=(10, 6))
    plt.plot(unattermp_daily_count.index, unattermp_daily_count.values, label='Proportion Unattempted')
    plt.xlabel('Chapter Day')
    plt.ylabel('Proportion of Unattempted Answers')
    plt.title('Trends in Unattempted Answers')
    plt.grid(True)
    plt.axhline(y=threshold, color='red', linestyle='--', label='Threshold')
    plt.legend()
    plt.show()
identify_spike_days(df) 


# Result:
#     
#     1.Day 11 to  13 unattempter rates are high and similarly day 27 to 28 students doesn't attempt the question may due to difficulty in chapters or teaching styles or due to month end bordem
#     
#     2.Through out the month the unattempted rate is above the threshold .The unattended answer rate appears to be relatively low overall, staying below 0.3 throughout the measured days.
#     

# In[ ]:





# 3.Random Performance Fluctuations: Spot patterns where student answers reflect randomness, suggesting days when students struggled significantly with the material.

# In[13]:


import seaborn as sns
def identify_random_performance_days(df):
    daily_answers=df.groupby(['Chapter','Day','Student ID'])['Status'].value_counts().reset_index()
    
    min_questions = 30
    filtered_df = daily_answers[
    (daily_answers['Status'].isin(['Correct', 'Incorrect'])) &
    (daily_answers['count'] >= min_questions)]

# Calculate daily_filtered_count with value counts and percentage
    daily_filtered_count = filtered_df.groupby(['Chapter', 'Day', 'Student ID'])['Status'].value_counts().reset_index()
    daily_filtered_count = daily_filtered_count.assign(percentage=daily_filtered_count['count'] / daily_filtered_count['count'].sum() * 100)
    

    
    
    
    
    
 
    ax = sns.scatterplot(
    x="Student ID",
    y="Day",
    hue="Status",
    size=(filtered_df['count'] / filtered_df['count'].sum())*100,
    data=daily_filtered_count)
    ax.set_title(" Student Performance day basis % count ")
    plt.show()

    ax2 = sns.scatterplot(
    x="Student ID",
    y="Day",
    hue="Status",
    size='percentage',
    data=daily_filtered_count)
    ax2.set_title(" Student Performance day basis ")
    plt.show()


    ax3 = sns.scatterplot(
    x="Student ID",
    y="Chapter",
    hue="Status",
    size='percentage',
    data=daily_filtered_count)
    ax3.set_title(" Student Performance Chapter basis overall percentge ")
    plt.show()


    ax4 = sns.scatterplot(
    x="Student ID",
    y="Chapter",
    hue="Status",
    size=(filtered_df['count'] / filtered_df['count'].sum())*100,
    data=daily_filtered_count)
    ax4.set_title("Chapter wise Student Performance with % count")

    plt.show()
identify_random_performance_days(df)


# Chapter 3: Chapter 3 seems to have consistently lower scores compared to other chapters. This suggests that students might find the material in Chapter 3 more challenging.
# Chapter 1 and 2: Scores in Chapters 1 and 2 appear to be spread across a wider range, with some students performing well and others performing poorly. This could indicate that these chapters might have a mix of easier and more difficult topics.
# Chapter 4 and 5: Scores in Chapters 4 and 5 seem to be generally higher and more clustered together. This suggests that students might find the material in these chapters easier to understand.
# Day-to-Day Performance:
# 
# Overall Trend: There appears to be a slight upward trend in average scores across days. This could suggest a learning effect, where students are gradually improving their understanding of the material.
# Variations: However, there are also variations in scores within each chapter across days. For example, in Chapter 1, scores seem to be lower on Day 2 compared to Day 3 and Day 4. This could be due to several factors, such as:
# Difficulty level of topics covered on those days.
# Student effort levels or engagement.
# External events that might have impacted performance (e.g., absences, school assemblies).

# Inference: Based on Student performance on day and chapter wise
# 
#     1.Most of the student got incorrect answer 8 and 9 th day.
#     
#     2.We can see few student with Id 1 and 3 have maximum days with correct answers for all the chapters
#     
#     3.Student Id with 6 and 9 faced incorrect answer  in chapter 2 and 3 

# Pattern Recognition:
# 
# 1.Determine if there is a learning curve visible where students start with poorer performance at the beginning of each chapter but improve as they become more familiar with the content.

# In[14]:


def identify_learning_curve(data):    
    average_scores = (df[df["Status"] == "Correct"].groupby(["Student ID", "Chapter"])["Day"].count().reset_index(name="Average Score")
                      .merge(df[["Student ID", "Chapter", "Day"]], how="left").fillna(0)  )
    

  # Calculate daily average score per chapter
    daily_chapter_average = (
      average_scores.groupby(["Chapter", "Day"])["Average Score"].mean().reset_index())

  # Plot average score vs Day with hue as Chapter
    sns.relplot(
      x="Day",
      y="Average Score",
      hue="Chapter",
      kind="line",
      data=average_scores,
      size="Student ID",)
    plt.title("Average Score by Day and Chapter")
    plt.show()

  # Plot daily chapter average vs Day
    sns.lineplot(
      x="Day",
      y="Average Score",
      hue="Chapter",
      data=daily_chapter_average,
      style="Chapter",)
    plt.title("Daily Average Score per Chapter")
    plt.show()

# Example usage (assuming your data is loaded into a DataFrame called 'df')
identify_learning_curve(df)


# 1.Chapter Comparison: For a specific day (e.g., Day 4), Chapter 2 appears to have a lower average score compared to Chapters 1, 3, and potentially 10. Similarly, for Day 7, Chapter 3 seems to have a lower average score. This suggests that students might find the material covered in these chapters more challenging.
# 2.Day-to-Day Variations: The average score varies across days for each chapter. For example, in Chapter 1, the score appears to be higher on Day 4 compared to Day 2. This could be due to several factors, such as difficulty of the material covered on those days, student effort levels, or external events.
# 3.Days with lower average scores (e.g., Day 2, Day 3, Day 6) might warrant further investigation to understand why students struggled more on those days.
# 4.Chapters with consistently lower average scores might indicate topics that require additional explanation or different teaching approaches.
# 5.Day 1 seems to have a higher average score compared to Day 2. Day 4 appears to have a higher average score than Day 3, and Day 8 has a higher average score than Day 7. This pattern suggests there might be some learning effect or improvement over time, but it's not consistent across all chapters.
# 6.Day 1 seems to have a higher average score compared to Day 2. This could be due to various reasons, such as a practice session on Day 1 or an easier topic being covered.
# Day 3 appears to have a lower average score than Day 4. This could be due to a more complex topic being introduced on Day 3.
#     

# 2.Assess whether performance varies significantly from the start to the end of each chapter, suggesting review or fatigue effects.

# In[15]:


daily_scores = (
    df.groupby(["Student ID", "Chapter", "Day"])["Status"]
    .apply(pd.Series.value_counts)
    .fillna(0)
    .unstack(fill_value=0)
)


# In[16]:


daily_scores["total"]=daily_scores["Incorrect"]+daily_scores["Correct"] 
daily_scores["Proportion Correct"]=daily_scores["Correct"]/daily_scores["total"]

 


# In[17]:


sns.relplot(
      x="Day",
      y="Proportion Correct",
      size="Chapter",
      
      kind="line",
      data=daily_scores,
  )
plt.title("Proportion Correct by Day and Chapter")
plt.show()


# In[18]:


import plotly.express as px
daily_scores["Total Answers"] = daily_scores["Correct"] + daily_scores["Incorrect"]

# Structure for treemap (parent-child relationships)
daily_scores_tree = daily_scores.pivot_table(
    values="Proportion Correct", index=["Chapter", "Day"], columns="Student ID", aggfunc=sum
)
daily_scores_tree = daily_scores_tree.stack().reset_index(name="Proportion Correct")

# Create treemap with plotly express
fig = px.treemap(
    daily_scores_tree,
    path=["Chapter", "Day", "Student ID"],
    values="Proportion Correct",
    color="Proportion Correct",
    color_continuous_scale="Viridis",  # Adjust color scale as needed
)
fig.update_layout(title="Proportion Correct by Chapter, Day, and Student")
fig.show()


# 
# inference:
# 1.Across days 7, 8, 9, 10, 26, 27, and 25, several students in Chapters 2 and 4 exhibited a performance range of 20% to 40% correct answers. This may indicate areas of difficulty that require further attention."
# 2.Chapter 3 showed peaks in student performance on days 17, 18, 19, 20, 22, and 24, with high percentages of correct answers. In contrast, day 21 saw a notable decline, with scores averaging around 40-50%, suggesting a potential difficulty spike for Chapter 3 on that day.
# 3.Our analysis of Chapter 2 reveals a low attempt rate for correct answers, suggesting initial difficulty. However, a positive trend emerges with a gradual increase in successful attempts after day 10."
# 4.Performance dips are observed even among high performers (students 1-4) towards the end of the month."
# 5.Chapter 2 appears to have a lighter color palette compared to Chapter 3, suggesting that students on average scored lower in Chapter 2
# 6.Improvement Over Time: Within some chapters, there appears to be a trend of improvement over time.  For example, in Chapter 3, the rectangles toward the right side (presumably later days) tend to be darker, indicating a higher proportion correct.

# In[ ]:




