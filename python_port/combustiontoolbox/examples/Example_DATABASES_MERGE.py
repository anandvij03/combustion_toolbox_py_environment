# -------------------------------------------------------------------------
# EXAMPLE: DATABASES MERGE
#
# @author: Anand V
# @adapted-from: Alberto Cuadra Lara 
#                 
# Last update June 2026
# -------------------------------------------------------------------------

# Import packages
from combustiontoolbox.databases.NasaDatabase.NasaDatabase import NasaDatabase

def main():
    # Get Nasa database
    print("Loading NASA database...")
    DB_NASA = NasaDatabase(thermoFile='thermo_NASA.inp')

    # Generate Burcat database using NASA9 format
    print("\nLoading Burcat database...")
    DB_BURCAT = NasaDatabase(thermoFile='thermo_millennium_2_thermoNASA9.inp')

    # Merge databases
    print("\nMerging databases...")
    DB = DB_NASA + DB_BURCAT

    # Display counts to verify successful merge
    print("\nMerge complete:")
    print(f"NASA database species count:   {DB_NASA.numSpecies}")
    print(f"Burcat database species count: {DB_BURCAT.numSpecies}")
    print(f"Merged database species count: {DB.numSpecies}")

if __name__ == "__main__":
    main()
