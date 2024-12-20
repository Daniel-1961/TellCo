import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df=pd.read_csv("C:\\Users\\lenovo\\OneDrive\\Desktop\\Copy of Week2_challenge_data_source(CSV).csv")
#print(df)
# TOP HANDSETS 
def data_overview():
    top_handsets = df['Handset Type'].value_counts().head(10)
    print("\nTop 10 Handsets:")
    print(top_handsets)
# TOP 3 HANDSET MANUFACTURERS
    top_manfacturers=df["Handset Manufacturers"].value_counts().head(3)
    print("\Top three handset manufacturers are")
    print(top_manfacturers)

#plot of top ten handsets

    plt.figure(figsize=(10, 6))
    sns.barplot(x=top_handsets.values, y=top_handsets.index, palette='viridis')
    plt.title('Top 10 Handsets Used by Customers')
    plt.xlabel('Number of Users')
    plt.ylabel('Handset Model')
    plt.show()

#TOP MANUFACTURERS 
    top_manfacturers=df['Handset Manufacturer'].value_counts().head(3)

#PLOT OF TOP MANFATCURERS

    plt.figure(figsize=(8, 6))
    top_manfacturers.plot(kind='bar', color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    plt.title('Top 3 Handset Manufacturers')
    plt.xlabel('Manufacturer')
    plt.ylabel('Number of Users')
    plt.show()

#TOP HNADSETS PER MANFACTURERS

    top_5_per_manufacturer = {}
    for manufacturer in top_manfacturers.index:
        top_5 = df[df['Handset Manufacturer'] == manufacturer]['Handset Type'].value_counts().head(5)
        top_5_per_manufacturer[manufacturer] = top_5

# top_5 Handset Type per Top_3 manufacturers
    for manufacturer, handset in top_5_per_manufacturer.items():
        print("\nTop five hansets for{manufacturer}:")
        print(handset)
#plot for 5 handsets in top three manufacurers

    plt.figure(figsize=(8,6))
    sns.barplot(x=handset.value,y=handset.index,palette='coolwarm')
    plt.title(f'Top 5 Handsets for {manufacturer}')
    plt.xlabel('Number of Users')
    plt.ylabel('Handset Type')
    plt.show()

