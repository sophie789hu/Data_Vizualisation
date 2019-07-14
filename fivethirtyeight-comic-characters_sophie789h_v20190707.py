import re
from datetime import datetime

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import squarify
from sklearn_pandas import CategoricalImputer

'''
Project started on 20th of May 2019 - by sophie789hu

In the area of the #meetoo, the question around gender equity araises (again) for the greatest interest of Wmen.
Recent concerns have been adressed regarding our skewed vision of genders.
Media, society, family... each of these factors and even more affect our jugdment against Women. Those contribute to 
the sustainment of gender stereotypes and prejudices, inherited by our Judeo-Christian culture.

Isn't it interesting to notice that the gender inequality starts from the omni-presence of male characters in our surrounding?
And even in the Comics literacy.

The brief analysis below does not aim to blaim DC and Marvel but more to encourage them to create more female characters in their stories.

'''

# IMPORT OF DC AND MARVEL DATASETS
df_dc = pd.read_csv("dc-wikia-data.csv")
df_marvel = pd.read_csv("marvel-wikia-data.csv")


# DESCRIPTION OF DC and MARVEL DATASETS
def DescribeDataframe(df_dc, df_marvel):
    """ Describe dataframe """
    for df in [df_dc, df_marvel]:
        print(df.info())
        print(df.head())
        print(df.describe())


DescribeDataframe(df_dc, df_marvel)

"""
The DC dataset contains 6896 entries such as each entry represents one character.
Each character is characterized by 13 variables or columns, such as:
1 - "page_id" of type integer, corresponds to the unique identifier of the character's page
2 - "name" of type object, corresponds to the unique name of the character. Most of the time,
   - if the character is a super-hero: "nickname" + "civil first and last name" in brankets (ex: "Batman (Bruce Wayne)"")
   - if the character is not a super-hero: "civil first and last name" + "universe" in brankets (ex: "James Gordon (New Earth)")
   The universes are "Earth-Two", "Earth-One" or "New Earth".
3 - "urlslug" of type object, corresponds to the unique url of the character within the wiki in "dc.fandom.com".
4 - "ID" of type object, indicates the identity status of the character: "Secret Identity", "Public Identity" or "Identity Unknown".
5 - "ALIGN" of type object, indicates if one character is good ("Good Characters"), bad ("Bad Characters"), neutral ("Neutral Characters") or "Reformed Criminals".
6 - "EYE" of type object, indicates the eye color of the character. There are 17 options from "Blue Eyes" to "Orange Eyes". The option "Auburn Hair" may be an error.
7 - "HAIR" of type object, indicates the hair color of the character. There are 17 options from "Black Hair" to "Platinum Blond Hair".
8 - "SEX" of type object, indicates the gender of the character: "Male Characters", "Female Characters", "Genderless Characters"
9 - "GSM" of type object, indicates if the character is a gender or sexual minority: "Bisexual Characters" or "Homosexual Characters".
10 - "ALIVE" of type object, indicates if the character is alive ("Living Characters") or deceased ("Deceased Characters")
11 - "APPEARANCES" of type object, indicates the number of appareances of the character in comic books (as of Sep. 2, 2014)
12 - "FIRST APPEARANCE" of type object, indicates the year and the month of the character's first appearance in a comic book, if available
13 - "YEAR" of type float, indicates the year of the character's first appearance in a comic book, if available. The year in the column "YEAR" is equal
The Marvel dataset contains 16376 entries such as each entry represents one character.
Each character is characterized by 13 variables or columns, same as in the dataset df_dc.
"""


# DATA CLEANING


def AlignColumn(df_dc, df_marvel):
    """ Align column names """
    list_df = [df_dc, df_marvel]
    for df in list_df:
        df.columns = df.columns.str.upper()
        df.rename(columns={'ID': 'IDENTITY TYPE', 'SEX': 'GENDER', 'APPEARANCES': 'NUMBER OF APPEARANCES',
                           'FIRST APPEARANCE': 'DATE OF FIRST APPEARANCE', 'YEAR': 'YEAR OF FIRST APPEARANCE', 'ALIGN': 'TEAM'}, inplace=True)
    return df_dc, df_marvel


def CompareDataType(df_dc, df_marvel):
    """ Compare column data type """
    for col in df_dc.columns:
        if df_dc[col].dtype != df_marvel[col].dtype:
            print(col)
        else:
            print(col, ":", df_dc[col].dtype, "vs.", df_marvel[col].dtype)


def AlignData(df_dc, df_marvel):
    """ Align data name and value """
    list_df = [df_dc, df_marvel]
    for df in list_df:
        # Align data type
        df["NUMBER OF APPEARANCES"] = df["NUMBER OF APPEARANCES"].apply(lambda line: int(line) if pd.isnull(line) is False else None)
        df["YEAR OF FIRST APPEARANCE"] = df["YEAR OF FIRST APPEARANCE"].apply(lambda line: str(int(line)) if pd.isnull(line) is False else None)
        # Align data value in column
        df.GENDER = df.GENDER.str.replace(" Characters", "")
        print(df.GENDER)
        # print(df.GENDER)
        df.GENDER.replace(r"Agender|Genderless|Genderfluid|Transgender", "Other", regex=True, inplace=True)
        df.TEAM = df.TEAM.str.replace(" Characters", "")
        df.TEAM = df.TEAM.str.replace("Reformed Criminals", "Neutral")
        df["IDENTITY TYPE"].replace(r"Identity Unknown|No Dual Identity|Known to Authorities Identity", "Other", regex=True, inplace=True)
        df["IDENTITY TYPE"] = df["IDENTITY TYPE"].str.replace(" Identity", "")
        df.ALIVE = df.ALIVE.str.replace(" Characters", "")
        df.ALIVE.replace("Living", "Alive", regex=True, inplace=True)
        df.ALIVE.replace("Deceased", "Dead", regex=True, inplace=True)
    return df_dc, df_marvel


def CompareYears_dc(df_dc):
    """ Compare year in "FIRST APPEARANCE" and "YEAR OF FIRST APPEARANCE" columns - in df_dc"""
    year_in_date_dc = df_dc["DATE OF FIRST APPEARANCE"].apply(lambda line: line.split(",")[0] if pd.isnull(line) is False else False)
    year_in_year_dc = df_dc["YEAR OF FIRST APPEARANCE"].apply(lambda line: line if pd.isnull(line) is False else False)
    print(year_in_date_dc[year_in_date_dc != year_in_year_dc])


def CompareYears_marvel(df_marvel):
    """ Compare year in "FIRST APPEARANCE" and "YEAR OF FIRST APPEARANCE" columns - in df_marvel"""
    year_in_date_marvel = df_marvel["DATE OF FIRST APPEARANCE"].apply(lambda line: line[-2:] if pd.isnull(line) is False else True)
    year_in_year_marvel = df_marvel["YEAR OF FIRST APPEARANCE"].apply(lambda line: str(int(line))[2:] if pd.isnull(line) is False else True)
    print(year_in_date_marvel[year_in_date_marvel != year_in_year_marvel])


def VerifyUnicity(df_dc, df_marvel):
    """ Verify if the unique values are not shared by Marvel and DC """
    sharedPageIds = df_dc[df_dc["PAGE_ID"].isin(df_marvel["PAGE_ID"]) == True]
    print(sharedPageIds)
    namesOfSharedPageIds = sharedPageIds[sharedPageIds["NAME"].isin(df_marvel["NAME"]) == True]
    print(namesOfSharedPageIds)
    sharedNames = df_dc[df_dc["NAME"].isin(df_marvel["NAME"]) == True]
    print(sharedNames)
    sharedUrls = df_dc[df_dc["URLSLUG"].isin(df_marvel["URLSLUG"]) == True]
    print(sharedUrls)


def MaintainUniqueness(df_dc, df_marvel):
    """ Maintain uniqueness by creating new ids and source column """
    df_dc["NEW_PAGEID"] = df_dc["PAGE_ID"].apply(lambda id: str(id)+"_dc")
    df_marvel["NEW_PAGEID"] = df_marvel["PAGE_ID"].apply(lambda id: str(id)+"_marvel")
    df_dc["COMICS"] = "DC"
    df_marvel["COMICS"] = "Marvel"
    return df_dc, df_marvel


def AppendDf(df_dc, df_marvel):
    """ Append df_dc and df_marvel """
    df_Comics = df_dc.append(df_marvel, sort=False).reset_index(drop=True)
    df_Comics.drop(["PAGE_ID", "URLSLUG", "DATE OF FIRST APPEARANCE"], axis=1, inplace=True)
    return df_Comics


def MissingValue(df_Comics):
    """ Count the missing values """
    list_missingValue = list()
    list_ratio = list()
    for col in df_Comics.columns:
        missingValue = df_Comics[col].isna().sum()
        ratio = "{:.2%}".format(missingValue/len(df_Comics))
        list_missingValue.append(missingValue)
        list_ratio.append(ratio)
    df_missingValue = pd.DataFrame({"column": df_Comics.columns, "number of missing values": list_missingValue, "ratio": list_ratio})
    df_missingValue.sort_values(by=['number of missing values'], ascending=False, inplace=True)
    df_missingValue.reset_index(drop=True, inplace=True)
    return df_missingValue


def PlotMissingValue(df_missingValue):
    """ Plot missing values """
    custom_palette = {}
    for value in range(len(df_missingValue)):
        if df_missingValue["number of missing values"][value] > 6000:
            custom_palette[df_missingValue["column"][value]] = 'lightsalmon'
        else:
            custom_palette[df_missingValue["column"][value]] = 'lightgrey'

    missingValue_fig = plt.figure(figsize=(8, 5))
    missingValue_ax = sns.barplot(x="column", y="number of missing values",
                                    palette=custom_palette,
                                    data=df_missingValue)
    missingValue_ax.set_title("Missing values per column in df_Comics", fontsize=12)
    for index, column in enumerate(missingValue_ax.patches):
        height = column.get_height()
        missingValue_ax.text(
            index,  # bar index (x coordinate of text)
            height+500,  # y coordinate of text
            '{}'.format(df_missingValue["ratio"][index]),  # y label
            ha='center',
            va='center',
            fontweight='light',
            size=10)
    missingValue_ax.set(xlabel=None, ylabel=None)
    missingValue_ax.set_xticklabels(labels=range(1, 14))
    sns.despine(left=True, bottom=True)
    missingValue_fig.show()
    return missingValue_fig, missingValue_ax


def CleanMissingValue(df_Comics):
    """ Clean missing values """
    # Drop columns "EYE", "GSM" and "HAIR" (containing <25% missing values)
    df_Comics.drop(["EYE", "GSM", "HAIR"], axis=1, inplace=True)
    # Drop "ALIVE" and "YEAR OF FIRST APPEARANCE" missing values (containing <4% missings values)
    df_Comics.dropna(subset=["ALIVE", "YEAR OF FIRST APPEARANCE"], inplace=True)
    # Fill "NUMBER OF APPEARANCES" missing values with mean
    appearancesMean = df_Comics["NUMBER OF APPEARANCES"].mean()
    df_Comics["NUMBER OF APPEARANCES"] = df_Comics["NUMBER OF APPEARANCES"].fillna(appearancesMean)
    # Fill "GENDER", "IDENTITY TYPE" and "TEAM" missing values with most frequent value.
    imputer = CategoricalImputer()
    df_Comics["GENDER"] = imputer.fit_transform(df_Comics["GENDER"])
    df_Comics["IDENTITY TYPE"] = imputer.fit_transform(df_Comics["IDENTITY TYPE"])
    df_Comics["TEAM"] = imputer.fit_transform(df_Comics["TEAM"])
    return df_Comics


df_dc, df_marvel = AlignColumn(df_dc, df_marvel)
CompareDataType(df_dc, df_marvel)
df_dc, df_marvel = AlignData(df_dc, df_marvel)
CompareYears_dc(df_dc)
CompareYears_marvel(df_marvel)
VerifyUnicity(df_dc, df_marvel)  # 476 different characters share the same page_ids. Neither Marvel or DC created a character with the same name or with the same URLSLUG.
df_dc, df_marvel = MaintainUniqueness(df_dc, df_marvel)
df_Comics = AppendDf(df_dc, df_marvel)
df_missingValue = MissingValue(df_Comics)
missingValue_fig, missingValue_ax = PlotMissingValue(df_missingValue)
df_Comics = CleanMissingValue(df_Comics)  # Df_Comics has been cut by 0,04% from 22192 rows to 22291 rows.


# EXPLORATORY DATA ANALYSIS

def setAx(axs):
    for ax in axs:
        ax.set(xlabel=None, ylabel=None)
        ax.tick_params(labelsize=7)
    sns.despine(left=True, bottom=True)
    return

# I - Gender disparity in the Comics Industry
# Set the figure
genderDisparity_fig = plt.figure(figsize=(9, 5))
genderDisparity_spec = gridspec.GridSpec(ncols=2, nrows=2, figure=genderDisparity_fig)
genderDisparity_ax1 = genderDisparity_fig.add_subplot(genderDisparity_spec[0, 0])
genderDisparity_ax2 = genderDisparity_fig.add_subplot(genderDisparity_spec[0, 1])
genderDisparity_ax3 = genderDisparity_fig.add_subplot(genderDisparity_spec[1, :])
palette = ["lightblue", "pink", "moccasin"]
genderDisparity_fig.text(s="Gender inequality in Comic literacy is undeniable", x=0.09, y=0.95, fontsize=15)
genderDisparity_fig.suptitle("Female characters are three times less represented than male and it is not improving over time", x=0.5, y=0.93, fontsize=11)
# Subplot 1 - Gnder disparty of the population
ratioGender = ["{:.2%}".format(df_Comics.GENDER.value_counts()[gender]/len(df_Comics)) for gender in range(3)]
labelRatio = [df_Comics.GENDER.unique()[gender] + "\n (" + ratioGender[gender] + ")" for gender in range(3)]
countGender_ax = squarify.plot(sizes=df_Comics.GENDER.value_counts(),  label=labelRatio, color=palette, text_kwargs={'fontsize': 7}, ax=genderDisparity_ax1)
countGender_ax.axis('off')
# Subplot 2 - Gender disparty per Comics
genderPerComics_ax = sns.countplot(x="COMICS", hue="GENDER", data=df_Comics, palette=palette, ax=genderDisparity_ax2)
genderPerComics_ax.legend_.remove()
# Subplot 3 - Number of character created per year
creationPerComics = df_Comics.groupby(["YEAR OF FIRST APPEARANCE", "GENDER"]).count()[["NAME"]].reset_index()
creationPerComics["YEAR OF FIRST APPEARANCE"] = creationPerComics["YEAR OF FIRST APPEARANCE"].apply(int)
creationPerComics_ax = sns.lineplot(x="YEAR OF FIRST APPEARANCE", y="NAME", data=creationPerComics, hue="GENDER", palette=palette, ax=genderDisparity_ax3)
creationPerComics_ax.legend_.remove()
# Set the axes
setAx([countGender_ax, genderPerComics_ax, creationPerComics_ax])
plt.show()


# II - Representation of the chracters by role
# Set the figure - FIND TTITLE AND SUBTITLE
roleGender_fig, (roleGender_axe1, roleGender_axe2, roleGender_axe3) = plt.subplots(1, 3, figsize=(10, 3))
roleGender_fig.text(s="While distribution of role per gender is quite proportionated ", x=0.09, y=0.94, fontsize=15)
roleGender_fig.suptitle('It seems like Comic writers may perceive female characters as more likely to be a good person than male characters', x=0.5, y=0.92, fontsize=10)
palette2 = ["pink", "lightblue", "moccasin"]
# Subplot 1 - Gender disparity per ALIVE
genderIfAlive = df_Comics.groupby(["GENDER", "ALIVE"]).count()[["NAME"]].reset_index()
genderIfAlive.sort_values(["GENDER", "ALIVE"], ascending=[True, False], inplace=True)
genderIfAlive_label = genderIfAlive.ALIVE[:4].to_list()
genderIfAlive_index = genderIfAlive.NAME[:4].to_list()
explode = (0.1, 0.15, 0, 0)
colors_genderIfAlive = ["pink", "pink", "lightblue", "lightblue"]
roleGender_axe1.pie(genderIfAlive_index, explode=explode, labels=genderIfAlive_label, autopct='%1.1f%%', startangle=90,
                    colors=colors_genderIfAlive, wedgeprops={"edgecolor": "1", 'linewidth': 1}, textprops={'fontsize': 7})
# Subplot 2 - Gender disparity per IDENTITY TYPE
genderPerIdentity = df_Comics.groupby(["GENDER", "IDENTITY TYPE"]).count()[["NAME"]].reset_index()
genderPerIdentity.sort_values(["GENDER", "IDENTITY TYPE"], ascending=[True, False], inplace=True)
genderPerIdentity_ax = sns.barplot(x="IDENTITY TYPE", y="NAME", data=genderPerIdentity, hue="GENDER", ax=roleGender_axe2, palette=palette2)
genderPerIdentity_ax.legend_.remove()
# Subplot 3 - Gender disparity per TEAM 
genderPerTeam = df_Comics.groupby(["GENDER", "TEAM"]).count()[["NAME"]].reset_index()
genderPerTeam.ticks = range(0,3)
genderPerTeam.names = genderPerTeam.TEAM.unique()
bars1 = genderPerTeam[genderPerTeam.GENDER == "Female"].NAME.tolist()
bars2 = genderPerTeam[genderPerTeam.GENDER == "Male"].NAME.tolist()
bars3 = genderPerTeam[genderPerTeam.GENDER == "Other"].NAME.tolist()
bars1_2 = np.add(bars1, bars2).tolist()
roleGender_axe3.bar(genderPerTeam.names, bars1, color="pink", edgecolor='white')
roleGender_axe3.bar(genderPerTeam.names, bars2, bottom = bars1, color="lightblue", edgecolor='white')
roleGender_axe3.bar(genderPerTeam.names, bars3, bottom=bars1_2, color="moccasin", edgecolor='white')
# Set the axes
setAx([roleGender_axe1, roleGender_axe2, roleGender_axe3])
plt.show()


# III - Analysis focused on the top 50 most popular Comics characters
# The female chracter with the highest number of appearances if Susan Storm ranks #13...
top50_df_fig = plt.figure(figsize=(10, 6))
top50_df_spec = gridspec.GridSpec(ncols=3, nrows=3, figure=top50_df_fig)
top50_df_ax1 = top50_df_fig.add_subplot(top50_df_spec[0, :2])
top50_df_ax2 = top50_df_fig.add_subplot(top50_df_spec[1, :2])
top50_df_ax3 = top50_df_fig.add_subplot(top50_df_spec[2, :2])
top50_df_ax4 = top50_df_fig.add_subplot(top50_df_spec[0, 2])
top50_df_ax5 = top50_df_fig.add_subplot(top50_df_spec[1, 2])
top50_df_ax6 = top50_df_fig.add_subplot(top50_df_spec[2, 2])
top50_df_fig.text(s="Gender disparity among the top 50 is quite representative of the population", x=0.1, y=0.93, fontsize=15)
top50_df_fig.suptitle("While repartition of roles is quite even, more popular male characters are created than female characters", x=0.47, y=0.92, fontsize=10)
# Subplot 1 - Distribution per number of appeareances
df_Comics["bins_NbOfAppearances"] = pd.cut(df_Comics["NUMBER OF APPEARANCES"], np.arange(0, 4500, 1000))
top50_df = df_Comics.nlargest(50, "NUMBER OF APPEARANCES")
top50_df_Count_ax = sns.countplot(x="bins_NbOfAppearances", hue="GENDER", data=top50_df, palette=palette, ax=top50_df_ax1)
top50_df_Count_ax.set(xlabel=None, ylabel=None, xticklabels= [1000, 2000, 3000, 4000])
# Subplot 2 - Corrolation betwwen number of appeareances and year of creation _ male
subset_Top50_Male = top50_df[top50_df["GENDER"] == "Male"]
df_appearanceCorrolation_Top50_Male = pd.crosstab(subset_Top50_Male["bins_NbOfAppearances"], subset_Top50_Male["YEAR OF FIRST APPEARANCE"])
df_appearanceCorrolation_Top50_Male.sort_values(by=["bins_NbOfAppearances"], ascending=True)
top50_maleAppearance_ax = sns.heatmap(df_appearanceCorrolation_Top50_Male, cmap="GnBu", ax=top50_df_ax2, linewidths=.5, cbar=False)
top50_maleAppearance_ax.set(xlabel=None, ylabel=None, yticklabels=[1000, 2000, 3000, 4000])
top50_maleAppearance_ax.tick_params(axis='both', which='major', labelsize=8, rotation=0)
# Subplot 3 - Corrolation betwwen number of appeareances and year of creation _ female
subset_Top50_Female = top50_df[top50_df["GENDER"] == "Female"]
df_appearanceCorrolation_Top50_Female = pd.crosstab(subset_Top50_Female["bins_NbOfAppearances"], subset_Top50_Female["YEAR OF FIRST APPEARANCE"])
df_appearanceCorrolation_Top50_Female.sort_values(by=["bins_NbOfAppearances"], ascending=True)
top50_femaleAppearance_ax = sns.heatmap(df_appearanceCorrolation_Top50_Female, cmap="RdPu", ax=top50_df_ax3, linewidths=.5, cbar=False)
top50_femaleAppearance_ax.set(xlabel=None, ylabel=None, yticklabels=[1000, 2000])
top50_femaleAppearance_ax.tick_params(axis='both', which='major', labelsize=8, rotation=0)
# Subplot 4 - Gender disparity per ALIVE
top50_Alive = pd.crosstab(top50_df["ALIVE"].count(), [top50_df["ALIVE"], top50_df["GENDER"]])
top50_Alive_ratio = ["{:.2%}".format(float((top50_Alive.iloc[0][gender]/top50_Alive.index.values[0]))) for gender in range(4)]
top50_Alive_index = ["Alive", "Alive", "Dead", "Dead"]
top50_Alive_label = [top50_Alive_index[gender] + "\n (" + top50_Alive_ratio[gender] + ")" for gender in range(4)]
top50_Alive_ax = squarify.plot(sizes=list(top50_Alive.values[0]), label=top50_Alive_label, color=["pink","lightblue","lavenderblush","powderblue"], text_kwargs={'fontsize': 7}, ax=top50_df_ax4)
top50_Alive_ax.axis('off')
# Subplot 5 - Gender disparity per IDENTITY TYPE
top50_IdentityType = pd.crosstab(top50_df["IDENTITY TYPE"].count(), [top50_df["IDENTITY TYPE"], top50_df["GENDER"]])
top50_IdentityType_ratio = ["{:.2%}".format(float((top50_IdentityType.iloc[0][gender]/top50_IdentityType.index.values[0]))) for gender in range(6)]
top50_IdentityType_index = ["Other", "Other", "Public", "Public", "Secret", "Secret"]
top50_IdentityType_label = [top50_IdentityType_index[gender] + "\n (" + top50_IdentityType_ratio[gender] + ")" for gender in range(6)]
palette3=["lavenderblush","powderblue","pink","lightblue","lightpink","lightskyblue"]
top50_IdentityType_ax = squarify.plot(sizes=list(top50_IdentityType.values[0]), color=palette3, label=top50_IdentityType_label, text_kwargs={'fontsize': 7}, ax=top50_df_ax5)
top50_IdentityType_ax.axis('off')
# Subplot 6 - Gender disparity per TEAM 
top50_Team = pd.crosstab(top50_df["IDENTITY TYPE"].count(), [top50_df["TEAM"], top50_df["GENDER"]])
top50_Team_ratio = ["{:.2%}".format(float((top50_Team.iloc[0][gender]/top50_Team.index.values[0]))) for gender in range(3)]
top50_Team_index = ["Good", "Good", "Neutral"]
top50_Team_label = [top50_Team_index[gender] + "\n (" + top50_Team_ratio[gender] + ")" for gender in range(3)]
palette3=["pink","lightblue","powderblue"]
top50_Team_ax = squarify.plot(sizes=list(top50_Team.values[0]), color=palette3, label=top50_Team_label, text_kwargs={'fontsize': 7}, ax=top50_df_ax6)
top50_Team_ax.axis('off')
# Set the axes
setAx([top50_df_ax1, top50_df_ax2, top50_df_ax3, top50_df_ax4, top50_df_ax5, top50_df_ax6])
for ax in [top50_df_Count_ax, top50_Alive_ax, top50_IdentityType_ax, top50_Team_ax]:
    ax.legend_.remove()
plt.show()


# IV - EXPORT PLOTS
def SavePlots(missingValue_fig, title):
    """ Save plot into images """
    for name in [missingValue_fig]:
        today = datetime.now().strftime("%Y_%m_%d_%H-%M")
        titleName = title +"_"+today+".png"
        print(titleName)
        name.savefig(titleName)


SavePlots(missingValue_fig,"missingValue")
SavePlots(genderDisparity_fig,"genderDisparity")
SavePlots(roleGender_fig, "roleGender")
SavePlots(top50_df_fig, "top50_df")