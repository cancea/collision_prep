import numpy as np
import pandas as pd
import random
import math

REQUIRED_COLUMNS = [
    'Year', 'Geography', 'Collision Severity',
    'Heavy Vehicle', 'Speed', 'Impaired', 'Highway',  'Pedestrian', 'Multi-Vehicle',
    'Number of Collisions',
    'Number of Vehicles: None', 'Number of Vehicles: Light', 'Number of Vehicles: Moderate',
    'Number of Vehicles: Severe', 'Number of Vehicles: Demolished',
    'Number of People: No Injury', 'Number of People: Injury - Minimal', 'Number of People: Injury - Minor',
    'Number of People: Injury - Major', 'Number of People: Fatality',
    ]

numeric_columns = [n for n in REQUIRED_COLUMNS if n.startswith('Number')]
index_columns = [n for n in REQUIRED_COLUMNS if not n.startswith('Number')]


def rand_round(x):
    """  Randomly round up or down to the nearest integer based on distance from the nearest integer.

         This is useful when large sets of numbers may be rounded down, but the total should be maintained.
         For example, 0.1 will return 0 for 9 out of 10 call and 1 for the remaining one (on average).
    """
    #return round(x)
    f = math.floor(x)
    r = x - f
    return f+1 if random.random() < r else f



def check_summary(summary):
    # Perform some cross checks
    check_fatalities = summary[(summary['Collision Severity'] == 'Fatality')
                               & (summary['Number of People: Fatality'] == 0)
                               & (summary['Number of Collisions'] > 0) ]
    check_non_fatalities = summary[(summary['Collision Severity'] != 'Fatality')
                                   & (summary['Number of People: Fatality'] != 0)]
    check_pdo = summary[(summary['Collision Severity'] == 'Property Damage Only')
                        & ((summary['Number of People: Fatality'] != 0)
                           |(summary['Number of People: Injury - Minimal'] != 0)
                           |(summary['Number of People: Injury - Minor'] != 0)
                           |(summary['Number of People: Injury - Major'] != 0))]


    if len(check_fatalities):
        print(check_fatalities[['Geography', 'Collision Severity', 'Number of Collisions', 'Number of Vehicles: Demolished', 'Number of People: Fatality']])
    if len(check_non_fatalities):
        print(check_non_fatalities)

    assert len(check_fatalities) == 0, ['Fatal collision, but no person with fatality']
    assert len(check_non_fatalities) == 0, ['Non-Fatal collision, but person with fatality']
    assert len(check_pdo) == 0, ['PDO collision, but person with injury or fatality']


def parse_ontario_data(args):
    print("Parsing Ontario data...")
    variables = {
        'D29': {'name': 'Heavy Vehicle',
                'groups': {True: [f'{x:02d}' for x in range(7, 21)] + ['98'] },
                'default': False
                 },
        'D15': {'name': 'Speed',
                'default':  False,
                'groups': {True: ['3', '4']}
                },
        'D16': {'name': 'Impaired',
                'default': False,
                'groups': {True: ['3', '4', '5']}
                },
        'B12': {'name': 'Highway',
                'default': False,
                'groups': {True: ['2']}
                },
        'B13': {'name': 'Collision Severity',
                'default': 'Property Damage Only',
                'groups': {'Fatality': ['1'],
                           'Injury': ['2']}
                },
         'I11': {'name': 'Pedestrian',
                'default': False,
                'groups': {True: [str(x) for x in range(0,10)]}
                },
         'B09': {'name': 'Multi-Vehicle',
                'default': True,
                'groups': {False: ['1']}
                },
        }


    df_injury = pd.read_csv(args.ontario_injury, low_memory=False, dtype={'B13': str, 'I07': str})
    # remove the extraneous PDO collisions and no-injury people from the injury dataset
    df_injury = df_injury[df_injury['B13'].isin(['1', '2']) & df_injury['I07'].isin(['1', '2', '3', '4'])]
    df_collisions = pd.read_csv(args.ontario_collisions, low_memory=False)
    common = ['B02', 'D07']
    print('common:', common)

    injury_cols = common + [c for c in df_injury.columns if c.startswith('I')]

    df = df_collisions.merge(df_injury[injury_cols], on=common, how='left')
    print("# records:", len(df))
    #raise "stop"
    # Rename the key columns
    for v in variables:
        df[v] = df[v].apply(str)
        column_name = variables[v]['name']
        default = variables[v]['default']
        remap = {}
        for k, tags in variables[v]['groups'].items():
            for tag in tags:
                remap[tag] = k
        df['tmp'] = df[v].apply(lambda x: remap.get(x, default))
        q = df[['B02', 'tmp']].groupby(['B02'], as_index=False).max().rename(columns={'tmp': column_name})
        df = df.merge(q, on='B02')


    key_columns = [v['name'] for v in variables.values()]

    # Extract the number of collisions

    collision_count = df[['B02'] + key_columns].groupby(['B02', 'Collision Severity'], as_index=False).max()
    collision_count['Number of Collisions'] = 1
    collision_count = collision_count[key_columns + ['Number of Collisions']].groupby(key_columns).sum()
    print("  # of collisions:", collision_count['Number of Collisions'].sum())
    # Extract the number of vehicles
    counts = {
        'Number of Vehicles: None': ('D48', ['0', '1']),
        'Number of Vehicles: Light': ('D48', '2'),
        'Number of Vehicles: Moderate': ('D48', '3'),
        'Number of Vehicles: Severe': ('D48', '4'),
        'Number of Vehicles: Demolished': ('D48', '5'),
        }

    vehicle_count = df[['B02', 'D01', 'D48'] + key_columns].groupby(['B02', 'Collision Severity', 'D01', 'D48'], as_index=False).max()
    for c in counts:
        k,v = counts[c]
        vehicle_count[k] = vehicle_count[k].apply(str)
        if type(v) == list:
            vehicle_count[c] = vehicle_count[k].apply(lambda x : x in v)
        else:
            vehicle_count[c] = vehicle_count[k] == v

    vehicle_count = vehicle_count[key_columns + list(counts.keys())].groupby(key_columns).sum()
    print("  # of vehicles  :", vehicle_count.sum().sum())

    # Extract the number of people
    counts = {
        'Number of People: No Injury': ('I07', '0'),
        'Number of People: Injury - Minimal': ('I07', '1'),
        'Number of People: Injury - Minor': ('I07', '2'),
        'Number of People: Injury - Major': ('I07', '3'),
        'Number of People: Fatality': ('I07', '4'),
        }

    people_count = df[['B02',  'I01', 'I07'] + key_columns].groupby(['B02', 'Collision Severity',  'I01'], as_index=False).max()

    for c in counts:
        k,v = counts[c]
        people_count[k] = people_count[k].apply(str)
        if type(v) == list:
            people_count[c] = people_count[k].apply(lambda x : x in v)
        else:
            people_count[c] = people_count[k] == v

    people_count = people_count[key_columns + list(counts.keys())].groupby(key_columns).sum()
    print("  # of people    :", people_count.sum().sum())

    summary = collision_count.merge(vehicle_count, left_index=True, right_index=True, how='left')
    summary = summary.merge(people_count, left_index=True, right_index=True, how='left')
    for n in numeric_columns:
        summary[n] = summary[n].fillna(0)

    if args.year:
        summary['Year'] = args.year
    summary['Geography'] = 'Ontario'
    summary = summary.reset_index()[REQUIRED_COLUMNS]
    print("  # of fatalities:", summary['Number of People: Fatality'].sum())

    check_summary(summary)
    return summary


def parse_national_data(args, ontario):
    """ Parse the national collision dataset, using Ontario data as a proxy to fill in missing data"""
    print("Parsing national data...")
    variables = {
        'V_TYPE': {'name': 'Heavy Vehicle',
                   'groups': {True:  ['7', '07', '8', '08', '9', '09', '21']},
                   'default': False
                 },

        'C_SEV': {'name': 'Collision Severity',
                'default': 'Property Damage Only',
                'groups': {'Fatality': ['1'],
                           'Injury': ['2']}
                },
         'P_PSN': {'name': 'Pedestrian',
                'default': False,
                'groups': {True: ['99']}
                },
         'C_VEHS': {'name': 'Multi-Vehicle',
                    'default': True,
                    'groups': {False: ['1']}
                },
        }

    df = pd.read_csv(args.national, low_memory=False)

    # Rename the key columns
    for v in variables:
        df[v] = df[v].apply(str)
        column_name = variables[v]['name']
        default = variables[v]['default']
        remap = {}
        for k, tags in variables[v]['groups'].items():
            for tag in tags:
                remap[tag] = k
        df['tmp'] = df[v].apply(lambda x: remap.get(x, default))
        q = df[['C_CASE', 'tmp']].groupby(['C_CASE'], as_index=False).max().rename(columns={'tmp': column_name})
        df = df.merge(q, on='C_CASE')


    # The national dataset does not distinguish between some causes, include property damage only
    # collisions, levels of vehicle damage, or severity of injury
    # The Ontario data is used as a proxy for these factors.

    # Extract the number of collisions
    key_columns = [v['name'] for v in variables.values()]

    collision_count = df[['C_CASE'] + key_columns].groupby(['C_CASE', 'Collision Severity'], as_index=False).max()
    #collision_count = collision_count.drop('C_CASE', axis=1)
    collision_count['Number of Collisions'] = 1
    collision_count = collision_count[key_columns + ['Number of Collisions']].groupby(key_columns).sum()
    print("  # of collisions:", collision_count['Number of Collisions'].sum())


    # Extract the number of vehicles
    df['C_VEHS'] = pd.to_numeric(df['C_VEHS'])
    vehicle_count = df[['C_CASE', 'C_VEHS'] + key_columns].groupby(['C_CASE', 'Collision Severity'] , as_index=False).max()
    vehicle_count = vehicle_count.rename(columns={'C_VEHS': 'Number of Vehicles'})#.drop('C_CASE', axis=1)
    vehicle_count = vehicle_count[key_columns + ['Number of Vehicles']].groupby(key_columns).sum()
    print("  # of vehicles  :", vehicle_count['Number of Vehicles'].sum())


    # Extract number of people
    counts = {
        'Number of People: No Injury': ('P_ISEV', '1'),
        'Number of People: Injury':    ('P_ISEV', '2'),
        'Number of People: Fatality':  ('P_ISEV', '3'),
        }

    people_count = df[['C_CASE', 'V_ID', 'P_ID', 'P_ISEV'] + key_columns].groupby(['C_CASE',  'V_ID', 'P_ID', 'P_ISEV'], as_index=False).max()
    for c in counts:
        k,v = counts[c]
        people_count[k] = people_count[k].apply(str)
        people_count[c] = people_count[k] == v

    people_count = people_count[key_columns + list(counts.keys())].groupby(key_columns).sum()
    print("  # of people    :", people_count.sum().sum())

    summary = collision_count.merge(vehicle_count, left_index=True, right_index=True)
    summary = summary.merge(people_count, left_index=True, right_index=True)

    print("  # of fatalities:", summary['Number of People: Fatality'].sum())
    print("  # of records: ", len(summary))

    print("Expanding national data to align with Ontario..")
    # Split the injuries by severity
    severity_levels = ['Number of People: Injury - Minimal',
                       'Number of People: Injury - Minor',
                       'Number of People: Injury - Major']
    severity_split = ontario[key_columns + severity_levels].groupby(key_columns).sum()
    severity_total = sum([severity_split[x] for x in severity_levels])
    for s in severity_levels:
        summary[s] = summary['Number of People: Injury'] * severity_split[s]/severity_total
    summary = summary.drop('Number of People: Injury', axis=1)

    # split the vehicles levels of damage
    damage_levels = [
        'Number of Vehicles: None',
        'Number of Vehicles: Light',
        'Number of Vehicles: Moderate',
        'Number of Vehicles: Severe',
        'Number of Vehicles: Demolished'
        ]

    damage_split = ontario[key_columns + damage_levels].groupby(key_columns).sum()
    damage_total = sum([damage_split[x] for x in damage_levels])
    for s in damage_levels:
        summary[s] = summary['Number of Vehicles'] * damage_split[s]/damage_total
    summary = summary.drop('Number of Vehicles', axis=1)


    summary = summary.reset_index()
    check_summary(summary)


    # Split the collisions by contributing factor based on Ontario result
    split_data    = ontario.drop(["Geography", "Year"], axis=1)
    split_totals = split_data.groupby(key_columns).transform('sum')
    for n in numeric_columns:
        split_data[n] = split_data[n]/split_totals[n]

    expanded = pd.merge(summary, split_data, how='left', on=key_columns, suffixes=("_total", "_split"))
    for n in numeric_columns:
        expanded[n] =  (expanded[n + "_total"] * expanded[n+"_split"]).fillna(0)
        expanded = expanded.drop([n + "_total", n+"_split"], axis=1)


    print("Checking after expanding...")
    print("  # of collisions:", expanded['Number of Collisions'].sum())
    print("  # of vehicles:", sum(expanded[x].sum() for x in REQUIRED_COLUMNS if x.startswith('Number of Vehicles')))
    print("  # of people:", sum(expanded[x].sum() for x in REQUIRED_COLUMNS if x.startswith('Number of People')))
    print("  # of fatalities:", expanded['Number of People: Fatality'].sum())

    # We assume that property damage only collisions scale with total collisions
    idx_columns2 = list(index_columns)
    idx_columns2.remove('Collision Severity')
    idx_columns2.remove('Year')
    idx_columns2.remove('Geography')

    ontario_injury = ontario[ontario['Collision Severity'] == 'Injury'].drop(['Collision Severity', 'Year', 'Geography'], axis=1).groupby(idx_columns2).sum()
    ontario_pdo_factor = ontario[ontario['Collision Severity'] == 'Property Damage Only'].drop(['Collision Severity', 'Year', 'Geography'], axis=1).groupby(idx_columns2).sum()

    expanded_fatality = expanded[expanded['Collision Severity'] == 'Fatality']
    expanded_injury = expanded[expanded['Collision Severity'] == 'Injury']
    expanded_pdo = expanded_injury.drop('Collision Severity', axis=1).groupby(idx_columns2).sum()
    for n in numeric_columns:
        expanded_pdo[n] = expanded_pdo[n] * (ontario_pdo_factor[n] / ontario_injury[n]).fillna(0)
    expanded_pdo = expanded_pdo.reset_index()
    expanded_pdo['Collision Severity'] = 'Property Damage Only'

    summary = pd.concat([expanded_fatality, expanded_injury, expanded_pdo[expanded.columns]], ignore_index=True)

    summary['Geography'] = 'Canada'
    summary['Year'] = args.year
    check_summary(summary)

    print("Splitting national data...")
    idx_columns3 = list(index_columns)
    idx_columns3.remove('Geography')
    national_indexed = summary.drop(['Geography'], axis=1).groupby(idx_columns3).sum()

    known_provinces =  {'Ontario': ontario}
    for p in known_provinces.values():
        provincial_indexed = p.drop(['Geography'], axis=1).groupby(idx_columns3).sum()
        for n in numeric_columns:
            national_indexed[n] = (national_indexed[n] - provincial_indexed[n]).fillna(0)

    summary =  national_indexed.reset_index()

    provincial_data = pd.read_csv(args.casualty)
    all_regions = ["Newfoundland and Labrador",
                   "Prince Edward Island",
                   "Nova Scotia",
                   "New Brunswick",
                   "Quebec",
                   "Ontario",
                   "Manitoba",
                   "Saskatchewan",
                   "Alberta",
                   "British Columbia",
                   "Yukon",
                   "Northwest Territories",
                   "Nunavut"]

    provincial_data = provincial_data[provincial_data['Region'].isin(all_regions)].copy()

    provincial_data["Fatalities"] = provincial_data['Fatalities Per 100,000 Population']*provincial_data['Population']/100000
    provincial_data["Injuries"] = provincial_data['Injuries Per 100,000 Population']*provincial_data['Population']/100000

    # remove the known provinces
    provincial_data = provincial_data[~provincial_data['Region'].isin(known_provinces)].copy()

    # Assume fatal collisions split as Fatalities
    provincial_data['Fatality Share'] = provincial_data["Fatalities"]/provincial_data["Fatalities"].sum()
    provincial_data['Injury Share'] = provincial_data["Injuries"]/provincial_data["Injuries"].sum()
    # Assume PDO collisions split as Injuries
    provincial_data['PDO Share'] = provincial_data["Injuries"]/provincial_data["Injuries"].sum()

    provinces = []
    for i, r in provincial_data.iterrows():
        print(" -", r['Region'])
        fatalities = summary[summary['Collision Severity'] == 'Fatality'].copy()
        for n in numeric_columns:
            fatalities[n] = (fatalities[n] * r['Fatality Share']).fillna(0)
        injuries = summary[summary['Collision Severity'] == 'Injury'].copy()
        for n in numeric_columns:
            injuries[n] =( injuries[n] * r['Injury Share']).fillna(0)
        pdo = summary[summary['Collision Severity'] == 'Property Damage Only'].copy()
        for n in numeric_columns:
            pdo[n] = (pdo[n] * r['PDO Share']).fillna(0)
        prov = pd.concat([fatalities, injuries, pdo], ignore_index=True)
        prov['Geography'] = r['Region']
        provinces.append(prov)

    provinces.extend( known_provinces.values() )

    summary = pd.concat(provinces, ignore_index=True, sort=True)
    print("ALL COLUMNS:", list(summary.columns))
    print("REQUIRED:", REQUIRED_COLUMNS)
    print("OTHER: ", set(summary.columns) - set(REQUIRED_COLUMNS))

    summary = summary.reset_index()[REQUIRED_COLUMNS]
    has_collision = summary['Number of Collisions'].apply(lambda x: 1 if x > 0 else 0)
    # Tidy up any minor inconsistencies that have arising due to fractional counts
    for n in numeric_columns:
        summary[n] = summary[n].apply(rand_round)*has_collision
    print("Checking after geography...")
    print("  # of records:", len(summary))
    print("  # of collisions:", summary['Number of Collisions'].sum())
    print("  # of vehicles:", sum(summary[x].sum() for x in REQUIRED_COLUMNS if x.startswith('Number of Vehicles')))
    print("  # of people:", sum(summary[x].sum() for x in REQUIRED_COLUMNS if x.startswith('Number of People')))
    print("  # of fatalities:", summary['Number of People: Fatality'].sum())

    # after rounding, it is possible that there could be some minor inconsistencies.
    # We clean those up to ensure the dataset is consistent


    def adjust_fatality(r):
        if r[0] != 'Fatality':
            return 0
        if r[0] == 'Fatality' and r[1] == 0:
            return 0
        return max(r[2], 1)


    def adjust_multivehicle(r):
        if r[0]:
            return r[1]
        return sum(r[2:])


    summary['Number of Collisions'] = summary[['Multi-Vehicle',
                                               'Number of Collisions',
                                               'Number of Vehicles: None',
                                               'Number of Vehicles: Light',
                                               'Number of Vehicles: Moderate',
                                               'Number of Vehicles: Severe',
                                               'Number of Vehicles: Demolished']].apply(adjust_multivehicle, axis=1)


    summary['Number of People: Fatality'] = summary[['Collision Severity', 'Number of Collisions', 'Number of People: Fatality']].apply(adjust_fatality, axis=1)


    print("Checking cleanup...")
    print("  # of records:", len(summary))
    print("  # of collisions:", summary['Number of Collisions'].sum())
    print("  # of vehicles:", sum(summary[x].sum() for x in REQUIRED_COLUMNS if x.startswith('Number of Vehicles')))
    print("  # of people:", sum(summary[x].sum() for x in REQUIRED_COLUMNS if x.startswith('Number of People')))
    print("  # of fatalities:", summary['Number of People: Fatality'].sum())

    check_summary(summary)
    return summary


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
                    prog = 'Collision Data Prep',
                    description = 'Aligns provincial collision data to standard social cost of collision model format. See the README for more details.',
                    )
    parser.add_argument("-y", "--year", required=True)
    parser.add_argument("--ontario-collisions", required=True, help="Ontario collision-level statistics from MTO, in CSV format")
    parser.add_argument("--ontario-injury", required=True, help="Ontario injury-level statistics from MTO, in CSV format")
    parser.add_argument("--national", required=True, help="National collision statistics from Transport Canada, in CSV format")
    parser.add_argument("--casualty", required=True, help="Provincial fatality and injury rates from Transport Canada, in CSV format")

    args = parser.parse_args()

    ontario = parse_ontario_data(args)
    ontario.to_csv('ontario_summary.csv', index=False)
    national = parse_national_data(args, ontario)

    output = "collision_summary_{}.csv".format(args.year)
    print("Saving to:", output)
    national.to_csv(output, index=False)
