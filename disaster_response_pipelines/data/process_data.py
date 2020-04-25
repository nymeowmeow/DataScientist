# import libraries
import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages, categories):
    """
        load data stored in csv file into dataframe, perform basic preprocessing
        to remove duplicate rows in each data frame, and then merge the message
        and categories data into a single data frame

        Args:
            messages - path to the messages csv file
            categories - path to the categories csv file

        Returns:
            df : dataframe which merge the data in messages and categories csv
    """
    # use pandas to read data into dataframe
    messages = pd.read_csv(messages)
    categories = pd.read_csv(categories)
    # preprocess, remove duplicates from the data frame
    messages = messages.drop_duplicates()
    categories = categories.drop_duplicates()
    #create a dataframe by joining the messages and categories data
    #only want to join if the id exists in both messages and categories, so we have to use inner join
    df = messages.merge(categories, on = 'id', how='inner')

    return df

def clean_data(df):
    """
        perform data clean before saving the data to sqlite db
        Arg:
           df - dataframe which data cleaning operation will be applied
        Returns:
           df - new data frame created from the input dataframe that has the categories expaned
                and duplicaets removed.
    """
    #split the values in categories column on the ';' character so that each value
    #becomes a separate column
    categories = df['categories'].str.split(';', expand=True)
    #use the first row of categories dataframe to create names for the categories data
    row = categories.iloc[0]
    category_names = row.apply(lambda x: x.split('-')[0])
    categories.columns = category_names
    #convert category values to 0 or 1
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].astype(str).str[-1]
    
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
    #the related columns has value 0,1,2 to be safe clip values to be 0,1 instead
    categories = categories.clip(0,1)
    #drop the original categories column from the dataframe
    df = df.drop(columns="categories")
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)
    #remove duplicates
    df = df.drop_duplicates()

    return df
 
def save_data(df, db_filename):
    """
        save the dataframe to sqlite db

        Args:
            df - dataframe to be saved to sqlite lite db
            db_filename - filename to be used to save sqlite db

        Returns:
            None
    """
    engine = create_engine('sqlite:///{}'.format(db_filename))
    df.to_sql('messages', engine, if_exists='replace', index=False) #write dataframe to db
    engine.dispose() #free resources

def main():
    """
       invoke the various functions defined above to read, process and stored the data to 
       sqlite db

       Args:
           messages file name   - path to the messages csv file
           categories file name - path to the categories csv file
           sqlite db name       - path to the output sqlite db file
    """
    #make sure we have the necssary input arguments
    if len(sys.argv) == 4:
        messagesPath, categoriesPath, dbpath = sys.argv[1:]

        print (f'loading message file: {messagesPath}, categories file: {categoriesPath} to sqlite db: {dbpath}')
        df = load_data(messagesPath, categoriesPath)
        print ('data cleaning before saving data to sqlite db')
        df = clean_data(df)
        print (f'save data to sqlite db: {dbpath}, row count: {len(df)}')
        save_data(df, dbpath)
    else:
        print ('Usage: python process_data.py <messages file name> <categories file name> <sqlite db name>') 

if __name__ == '__main__':
    main()
