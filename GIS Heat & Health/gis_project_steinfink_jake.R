
# Setup -------------------------------------------------------------------

library(sf)
library(tidyverse)
library(purrr)
library(tmap)
library(reshape)

# Data Read In ------------------------------------------------------------

df_econ <- 
  
  # reading in econ charcateristcs table
  
  st_read('data/ACS_Economic_Characteristics_DC_Census_Tract.geojson') %>%
  set_names(
    names(.) %>%
      tolower()) %>%
  st_transform(crs = 2248) %>%
  
  # columns came in with auto generated names - had to manually rename the
  # relevant ones
  
  dplyr::rename(
    unemployment_rate = dp03_0009pe,
    median_household_income = dp03_0062e,
    per_capita_income = dp03_0088e,
    percent_under_poverty_line_18p = dp03_0133p) %>%
  
  ##########
  # outdoor_work_p is the percentage of individuals working in construction and 
  # agriculture/hunting/fishery industries - health_insurance_p is the percent
  # of people with healthcare coverage
  ##########

  mutate(outdoor_work_p = (dp03_0034e + dp03_0033e)/dp03_0003e,
         health_insurance_p = dp03_0096e/dp03_0095e,
         private_health_insurance_p = dp03_0097e/dp03_0095e,
         public_health_insurance_p = dp03_0098e/dp03_0095e,
         geoid = as.numeric(geoid)) %>%
  select(geoid, aland, median_household_income, unemployment_rate, 
         outdoor_work_p:public_health_insurance_p) %>%
  arrange(geoid)

# Data containing Heat indices and related vars

df_heat <-
  st_read('data/Heat_Sensitivity_Exposure_Index.geojson') %>%
  set_names(
    names(.) %>%
      tolower()
  ) %>%
  st_transform(crs = 2248) %>%
  select(id2, totalpop:p_poc, p_disability, 
         asthma:p_treecover, p_impsurf:hsei) %>%
  dplyr::rename(geoid = id2) %>%
  arrange(geoid)

# Data containing a random sample of Washington DC Buildings

dc_buildings <- 
  st_read('data/dc_buildings.geojson') %>% 
  st_transform(crs = 2248)

# Data containing points of cooling centers 

cooling_centers <- 
  st_read('data/Cooling_Centers.geojson') %>%
  st_transform(crs = 2248)


# Cleaning --------------------------------------------------------


# Looking at summary stats of columns in both data frames to see if anything is
# abnormal 

list(df_econ, 
     df_heat)%>%
  purrr::map(
    ~summary(.)
  )

# There is 1 value of -9999 in hsei and hsi so this will be made a missing

df_heat[df_heat$hsei == -9999,c('hsei','hsi')] <- NA


# Creating master dataset------------------------------------------------

merged_df <- 
  df_heat %>%
  
  # Converting to tibble to remove class of shapefile and perform inner join
  
  as_tibble() %>%
  select(geoid:hsei) %>%
  inner_join(.,
             df_econ %>% 
               as_tibble(),
             by = 'geoid')%>%
  st_as_sf() %>%
  st_transform(crs = 2248) %>%
  
  # Adding population density variable
  
  mutate(
    pop_density = totalpop/aland)

# Taking merged df and adding distance to cooling centers as a variable

df <- 
  dc_buildings %>%
  
  # Add distance column
  
  mutate(
    distance_to_cool = 
      
      # st_distance to compute distance
      
      st_distance(.,
                  
                  # Must unite the cooling centers into one polygon 
                  
                  st_union(cooling_centers) %>%
                    
                    # convert back to sf
                    
                    st_sf(),
                  by_element = TRUE) %>%
      units::set_units("km") %>%
      as.numeric()) %>%
  
  
  # Spatial join to put the buildings into tracts
  
  st_join(merged_df,.) %>%
  
  # Convert to tibble to calculate average distance by census tract
  
  as_tibble() %>%
  group_by(geoid) %>%
  summarise(distance_to_cool = 
              mean(distance_to_cool)) %>%
  
  # Convert back to shapefile
  
  full_join(merged_df, ., by = 'geoid') %>%
  st_as_sf(crs = 2248)

# EDA ---------------------------------------------------------------------

## Function to remove duplicate information from correlation matrix

get_lower_tri<-function(cormat){
  cormat[upper.tri(cormat)] <- NA
  return(cormat)
}

## Making a correlation heatmap

df %>%
  as_tibble() %>%
  select(p_poc:public_health_insurance_p, pop_density:distance_to_cool) %>%
  cor(., use = "pairwise.complete.obs") %>%
  round(. , 2) %>%
  get_lower_tri() %>% 
  
  # Flatten matrix into plottable dataframe
  
  melt(., na.rm = TRUE) %>%
  ggplot(., aes(X1, X2, fill= value)) +
  geom_tile(color='white') +
  scale_fill_gradient2(low="blue", 
                       mid="white",
                       high="red",
                       na.value = 'white') +
  geom_text(aes(X1, X2, label = value), 
            color = "black",
            size = 3) +
  theme_minimal() +
  ggtitle('Correlation Matrix')+
  coord_equal() +
  labs(x="",y="",fill="Correlation") +
  theme(axis.text.x =
          element_text(size=10, 
                       angle=45, 
                       vjust=1, 
                       hjust=1, 
                       margin=margin(-3,0,0,0)),
        axis.text.y =
          element_text(size=10, 
                       margin =
                         margin(0,-3,0,0)),
        panel.grid.major = 
          element_blank()) 

# Summary Table by Income Bin ---------------------------------------------

## Summary table of income brackets by p_poc and other metric averages

summary(df$median_household_income)

make_percent <- function(x) {
  round(100*x,2)
}

df %>%
  as_tibble() %>%
  
  # Creating bins for median salary by census tract
  # Breaks are at 35K - the poverty line, 60K the bottom 75% cutoff,
  # 100K - the median, 135K - top 75% cutoff, and 135K+
  
  mutate(
    census_income_bracket = 
  cut(df$median_household_income,
      breaks =  c(0, 35000, 60000, 100000, 135000, 1000000),
      labels = c('< 35K','35K - 60K', '60K-100K',' 100K-135K', '135K+'),
      include.lowest = TRUE,
      right = FALSE)) %>%
  group_by(census_income_bracket) %>%
  
  # Summarize the means by census tract of each variable
  
  summarise(
    Count = n(),
    POC = round(mean(p_poc), 2),
    Disability = round(mean(p_disability),2),
    Asthma = round(mean(asthma), 2),
    Heart_Disease = round(mean(chd),2),
    HSI = round(100 * mean(hsi), 2),
    Public_Insurance = round( 100 * mean(public_health_insurance_p), 2),
    Unemployment = round(mean(unemployment_rate), 2),
    Outdoor_Work = round(100 * mean(outdoor_work_p),2)) %>%
  drop_na() %>%
  kableExtra::kbl() %>%
    kableExtra::kable_styling()
  
# Scatter plots of predictions --------------------------------------------

#POC % vs. HSI 

ggplot(df, aes(p_poc, hsi, color = 'red'))+
  geom_point() + 
  geom_smooth() + 
  theme_minimal()+
  labs(
    x = 'People of Color (%)',
    y = "Heat Sensitivity Index",
    title = 'Minority Population vs. HSI by DC Census Tract')+
  theme(legend.position = "none")

# POC % vs. HEI

ggplot(df, aes(p_poc, hei, color = 'red'))+
  geom_point() + 
  geom_smooth() + 
  theme_minimal()+
  labs(
    x = 'People of Color (%)',
    y = "Heat Exposure Index",
    title = 'Minority Population vs. HEI by DC Census Tract')+
  theme(legend.position = "none")

# POC % vs. income

ggplot(df, aes(median_household_income,p_treecover, color = 'red'))+
  geom_point() + 
  geom_smooth() + 
  theme_minimal()+
  labs(
    y = 'Treecover %',
    x = "Median Household Income",
    title = 'Median Income vs. Treecover')+
  theme(legend.position = "none", plot.title = element_text(hjust = .5))+
  scale_x_continuous(labels = c('0','50K', '100K', '150K', '200K', '250K'))

# POC % vs. asthma

ggplot(df, aes(p_poc, asthma, color = 'red'))+
  geom_point() + 
  geom_smooth() + 
  theme_minimal()+
  labs(
    x = 'People of Color (%)',
    y = "Prevalence of Asthma",
    title = 'Minority Population vs. Asthma Prevalence')+
  theme(legend.position = "none")

# POC % vs. obesity

ggplot(df, aes(p_poc, obesity, color = 'red'))+
  geom_point() + 
  geom_smooth() + 
  theme_minimal()+
  labs(
    x = 'People of Color (%)',
    y = "Prevalence of Obesity",
    title = 'Minority Population vs. Obesity Prevalence')+
  theme(legend.position = "none")

# POC vs Public Insurance %

ggplot(df, aes(p_poc, public_health_insurance_p, color = 'red'))+
  geom_point() + 
  geom_smooth() + 
  theme_minimal()+
  labs(
    x = 'People of Color (%)',
    y = "% Publically Insured",
    title = 'Minority Population vs. % Publically Insured')+
  theme(legend.position = "none")

# POC vs Distance to cool

ggplot(df, aes(p_poc, distance_to_cool, color = 'red'))+
  geom_point() + 
  geom_smooth() + 
  theme_minimal()+
  labs(
    x = 'People of Color (%)',
    y = "Average Distance to Cooling Center (Km)",
    title = 'Minority Population vs. Avg. Distance to Cooling Center')+
  theme(legend.position = "none")


# Static Maps  -------------------------------------------------------------

#Side by side of POC and HSI - why this pattern?

ggplot(data = df) + 
  geom_sf(aes(fill = p_poc)) + 
  theme_void()+
  scale_fill_continuous(name = "POC%", type = "viridis")+
  theme(legend.position = "left",
        legend.title = element_text(color = "white", size = 14),
        legend.text = element_text(color = "white", size = 10),
        plot.background = element_rect(fill = "black"))
  
ggplot(data = df) + 
  geom_sf(aes(fill = hsi)) + 
  theme_void()+
  scale_fill_continuous(name = "HSI", type = "viridis")+
  theme(legend.position = "left",
        legend.title = element_text(color = "white", size = 14),
        legend.text = element_text(color = "white", size = 10),
        plot.background = element_rect(fill = "black"))

# Seeing if this is due to heat exposure

ggplot(data = df) + 
  geom_sf(aes(fill = hei)) + 
  theme_void()+
  scale_fill_continuous(name = "HEI", type = "viridis")+
  theme(legend.position = "left",
        legend.title = element_text(color = "white", size = 14),
        legend.text = element_text(color = "white", size = 10),
        plot.background = element_rect(fill = "black"))
  
# Seeing if this is due to health by looking at asthma

ggplot(data = df) + 
  geom_sf(aes(fill = asthma)) + 
  theme_void()+
  scale_fill_continuous(name = "Asthma", type = "viridis")+
  theme(legend.position = "left",
        legend.title = element_text(color = "white", size = 14),
        legend.text = element_text(color = "white", size = 10),
        plot.background = element_rect(fill = "black"))

# Examine heart disease

ggplot(data = df) + 
  geom_sf(aes(fill = chd)) + 
  theme_void()+
  scale_fill_continuous(name = "Heart Disease", type = "viridis")+
  theme(legend.position = "left",
        legend.title = element_text(color = "white", size = 14),
        legend.text = element_text(color = "white", size = 10),
        plot.background = element_rect(fill = "black"))
  
# Examine obesity

ggplot(data = df) + 
  geom_sf(aes(fill = obesity)) + 
  theme_void()+
  scale_fill_continuous(name = "Obesity", type = "viridis")+
  theme(legend.position = "left",
        legend.title = element_text(color = "white", size = 14),
        legend.text = element_text(color = "white", size = 10),
        plot.background = element_rect(fill = "black"))

# What about Disability?

ggplot(data = df) + 
  geom_sf(aes(fill = p_disability)) + 
  theme_void()+
  scale_fill_continuous(name = "Disability%", type = "viridis")+
  theme(legend.position = "left",
        legend.title = element_text(color = "white", size = 14),
        legend.text = element_text(color = "white", size = 10),
        plot.background = element_rect(fill = "black"))
  
# Examine Income

ggplot(data = df) + 
  geom_sf(aes(fill = median_household_income)) + 
  theme_void()+
  scale_fill_continuous(name = "Median Income", 
                        type = "viridis",
                        breaks = c(50000, 100000, 150000, 200000, 250000),
                        labels = c('50K', '100K', "150K", "200K", "250K"))+
  theme(legend.position = "left",
        legend.title = element_text(color = "white", size = 14),
        legend.text = element_text(color = "white", size = 10),
        plot.background = element_rect(fill = "black"))

# Examine Public Health Insurance %

ggplot(data = df) + 
  geom_sf(aes(fill = public_health_insurance_p)) + 
  theme_void()+
  scale_fill_continuous(name = "Public Insurance %", type = "viridis")+
  theme(legend.position = "left",
        legend.title = element_text(color = "white", size = 14),
        legend.text = element_text(color = "white", size = 10),
        plot.background = element_rect(fill = "black"))
  
# Is it because of where you work (outdoor vs  indoor)

ggplot(data = df) + 
  geom_sf(aes(fill = outdoor_work_p)) + 
  theme_void()+
  scale_fill_continuous(name = "Outdoor Work", type = "viridis")+
  theme(legend.position = "left",
        legend.title = element_text(color = "white", size = 14),
        legend.text = element_text(color = "white", size = 10),
        plot.background = element_rect(fill = "black"))

# What about the built enironment

ggplot(data = df) + 
  geom_sf(aes(fill = p_treecover)) + 
  theme_void()+
  scale_fill_continuous(name = "Treecover", type = "viridis")+
  theme(legend.position = "left",
        legend.title = element_text(color = "white", size = 14),
        legend.text = element_text(color = "white", size = 10),
        plot.background = element_rect(fill = "black"))

# What about the distance to cooling centers

ggplot(data = df) + 
  geom_sf(aes(fill = distance_to_cool)) + 
  theme_void()+
  scale_fill_continuous(name = "Distance to Cool (Km)", type = "viridis")+
  theme(legend.position = "left",
        legend.title = element_text(color = "white", size = 14),
        legend.text = element_text(color = "white", size = 10),
        plot.background = element_rect(fill = "black"))

# Interactive Maps ---------------------------------------------------------

tmap_mode("view")

tm_shape(df,
         name = "Income")+
  tm_polygons("median_household_income",
              title = "Median Income") + 
  tm_shape(df,
           name = "POC")+
  tm_polygons("p_poc",
              title = "POC%") +
  tm_shape(df,
           name = "HSI") + 
  tm_dots("hsi",
          title = "HSI",
          palette = "viridis")+
  tm_shape(cooling_centers,
           name = "Cooling Centers")+
  tm_markers()
