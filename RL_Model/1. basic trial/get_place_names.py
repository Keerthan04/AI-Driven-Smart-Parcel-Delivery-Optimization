from geopy.geocoders import Nominatim
import pandas as pd
def get_individual_place_name(geolocator, lat, lon):
    location = geolocator.reverse((lat, lon), language='en', timeout=10)
    if location:
        return location.address
    else:
        return None
def get_place_names(df, lat_col='y', lon_col='x'):
    """
    Get place names from latitude and longitude using Nominatim.

    Args:
        df (pd.DataFrame): DataFrame containing latitude and longitude columns.
        lat_col (str): Name of the latitude column. Default is 'latitude'.
        lon_col (str): Name of the longitude column. Default is 'longitude'.

    Returns:
        pd.DataFrame: DataFrame with an additional 'place_name' column.
    """
    # Initialize the geolocator
    geolocator = Nominatim(user_agent="place_name_fetcher")
    df['place_name'] = df.apply(
        lambda row: get_individual_place_name(geolocator, row[lat_col], row[lon_col]), axis=1
    )
    return df
if __name__ == "__main__":
    df = pd.read_csv('./nodes.csv')
    df = get_place_names(df, lat_col='y', lon_col='x')
    df.to_csv('./nodes_with_place_names.csv', index=False)
    print("Place names added to DataFrame and saved to nodes_with_place_names.csv")
