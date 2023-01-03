# setup -----------------------------------------

library(shiny)
library(shinydashboard)
library(sf)
library(tidyverse)
library(purrr)
library(tmap)


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
  dc_buildings[sample(1:nrow(dc_buildings), 500), ] %>%
  
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


# Static Outputs in Server ------------------------------------------------

interactive_map <- 
  tm_basemap('Esri.WorldTopoMap')+
  tm_shape(df,
           name = "Income")+
  tm_polygons("median_household_income",
              title = "Median Income")+
  tm_shape(df,
           name = "POC")+
  tm_polygons("p_poc",
              title = "POC%") +
  tm_shape(df,
           name = "HSI") + 
  tm_dots("hsi",
          title = "HSI",
          palette = "viridis",
          size = .035)+
  tm_shape(cooling_centers,
           name = "Cooling Centers")+
  tm_markers(size = 0.75) 

static_plot1a <- 
  ggplot(df, aes(p_poc, hsi, color = 'red'))+
  geom_point() + 
  geom_smooth() + 
  theme_minimal()+
  labs(
    x = 'People of Color (%)',
    y = "Heat Sensitivity Index",
    title = 'Minority Population vs. HSI')+
  theme(legend.position = "none", 
        plot.title = element_text(hjust = 0.5, size = 14),
        axis.title.x = element_text(size = 12),
        axis.title.y = element_text(size = 12))

static_plot1b <- 
  ggplot(df, aes(p_poc, hei, color = 'red'))+
  geom_point() + 
  geom_smooth() + 
  theme_minimal()+
  labs(
    x = 'People of Color (%)',
    y = "Heat Exposure Index",
    title = 'Minority Population vs. HEI by DC Census Tract')+
  theme(legend.position = "none",
        plot.title = element_text(hjust = 0.5, size = 14),
        axis.title.x = element_text(size = 12),
        axis.title.y = element_text(size = 12))

static_plot2a <- 
  ggplot(df, aes(p_poc, asthma, color = 'red'))+
  geom_point() + 
  geom_smooth() + 
  theme_minimal()+
  labs(
    x = 'People of Color (%)',
    y = "Prevalence of Asthma",
    title = 'Minority Population vs. Asthma Prevalence')+
  theme(legend.position = "none",
        plot.title = element_text(hjust = 0.5, size = 14),
        axis.title.x = element_text(size = 12),
        axis.title.y = element_text(size = 12))

static_plot2b <-
  ggplot(df, aes(p_poc, median_household_income, color = 'red'))+
  geom_point() + 
  geom_smooth() + 
  theme_minimal()+
  labs(
    x = 'People of Color (%)',
    y = "Median Household Income",
    title = 'Minority Population vs. Median Income')+
  theme(legend.position = "none",
        plot.title = element_text(hjust = 0.5, size = 14),
        axis.title.x = element_text(size = 12),
        axis.title.y = element_text(size = 12))+
  scale_y_continuous(labels = 
                       c('0','50K', '100K', '150K', '200K', '250K'))

static_plot3a <- 
  ggplot(df, aes(p_poc, obesity, color = 'red'))+
  geom_point() + 
  geom_smooth() + 
  theme_minimal()+
  labs(
    x = 'People of Color (%)',
    y = "Prevalence of Obesity",
    title = 'Minority Population vs. Obesity Prevalence')+
  theme(legend.position = "none",
        plot.title = element_text(hjust = 0.5, size = 14),
        axis.title.x = element_text(size = 12),
        axis.title.y = element_text(size = 12))

static_plot3b <- 
  ggplot(df, aes(p_poc, distance_to_cool, color = 'red'))+
  geom_point() + 
  geom_smooth() + 
  theme_minimal()+
  labs(
    x = 'People of Color (%)',
    y = "Avg. Dist to CC (Km)",
    title = 'Minority Population vs. Dist to Cooling Center')+
  theme(legend.position = "none",
        plot.title = element_text(hjust = 0.5, size = 14),
        axis.title.x = element_text(size = 12),
        axis.title.y = element_text(size = 12))

static_plot4a <- 
  ggplot(df, aes(p_poc, public_health_insurance_p, color = 'red'))+
  geom_point() + 
  geom_smooth() + 
  theme_minimal()+
  labs(
    x = 'People of Color (%)',
    y = "% Publically Insured",
    title = 'Minority Population vs. % Publically Insured')+
  theme(legend.position = "none", 
        plot.title = element_text(hjust = 0.5, size = 14),
        axis.title.x = element_text(size = 12),
        axis.title.y = element_text(size = 12))

static_plot4b <- 
  ggplot(df, aes(p_poc, p_treecover, color = 'red'))+
  geom_point() + 
  geom_smooth() + 
  theme_minimal()+
  labs(
    x = 'People of Color (%)',
    y = "Treecover %",
    title = 'Minority Population vs. Treecover')+
  theme(legend.position = "none",
        plot.title = element_text(hjust = 0.5, size = 14),
        axis.title.x = element_text(size = 12),
        axis.title.y = element_text(size = 12))

data_table <- 
  df %>%
  as_tibble() %>%
  
  # Creating bins for median salary by census tract
  # Breaks are at 35K - the poverty line, 60K the bottom 75% cutoff,
  # 100K - the median, 135K - top 75% cutoff, and 135K+
  
  mutate(
    Census_Income_Bracket = 
      cut(
        df$median_household_income,
        breaks = c(0, 35000, 60000, 100000, 135000, 1000000),
        labels = 
          c('< 35K','35K - 60K', '60K-100K',' 100K-135K', '135K+'),
        include.lowest = TRUE,
        right = FALSE)) %>%
  group_by(Census_Income_Bracket) %>%
  
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
  drop_na()
  
# ui --------------------------------------------

ui <- 
  dashboardPage(
    
    dashboardHeader(title = 'Heat & Health in DC'),
    
    dashboardSidebar(
      sidebarMenu(
        
        menuItem(
          'Overview',
          tabName = 'overview',
          icon = icon('book')),
        menuItem(
          'Interactive Map',
          tabName = 'interactive_map',
          icon = icon('map')),
        
        menuItem('Map',
                 icon = icon('map'),
                 tabName = 'maps'),
        
        menuItem('Plots',
                 icon = icon('chart-line'),
                 tabName = 'charts'),
        
        menuItem('Table',
                 icon = icon('table'),
                 tabName = 'tables'),
        menuItem("Takeaway",
                 icon = icon('list-check'),
                 tabName = 'takeaway')
      ),
      hr()
    ),
    
    dashboardBody(
      tags$head(
        tags$link(
          rel = 'stylesheet',
          type = 'text/css',
          href = 'dashboard_styles.css'
        )
      ),
      
      tabItems(
        
        tabItem(
          tabName = 'overview',
          h2('Overview'),
          HTML("<p>The world is getting hotter. Heat wave seasons are getting 
               longer, more intense, and the effects are only starting to be 
               seen. To better prepare as the world changes, it is crucial to 
               understand who is vulnerable and why. In this app, we focus on 
               Washington, D.C. - an urban area of great racial diversity 
               (46% black, 37% white only, 11.5% Latino, 4.5% Asian, 1% other) 
               according to the <a href= 'https://www.census.gov/quickfacts/DC',
               target = '_blank'> U.S. Census Bureau</a>. Due to this racial 
               makeup, in addition to the known inequities in wealth in the 
               Nation's capital and its swampy climate, D.C. Census Tracts make 
               for a great case study to analyze. Much of the data for this 
               project is from <a href = 'https://opendata.dc.gov', 
               target = '_blank'> Open Data DC</a>, a repository for data
               maintained by the government.</p>"),
          p('Click the menu items in the sidebar menu on the left to explore 
            what this app can do!'),
          br(),
          br(),
          fluidRow(
            column(6, 
                   h3("Change in Temperature Over Time"),
                   HTML("<img src = 'https://static01.nyt.com/images/2020/04/23/learning/GlobalTemp2019GraphLN/GlobalTemp2019GraphLN-superJumbo.png',
                        alt = 'Graph Showing Heat Over Time', height = '425', 
                        width = '100%'"),
                   br(),
                   br(),
                   HTML("<p>Source: <a href = 'https://www.nytimes.com/2020/04/23/learning/whats-going-on-in-this-graph-global-temperature-change.html', 
                        target = '_blank'>
                        NYT</a></p>")
            ),
            column(6, 
                   h3("Heat Wave Season Length and Intensity Over Time"),
                   HTML("<img src = 'https://www.globalchange.gov/sites/globalchange/files/heat_waves_v1_052621.png',
                         alt = 'Graph showing Heat Waves Over Time', 
                        height = '425', width = '100%'"),
                   br(),
                   br(),
                   HTML("<p>Source: <a href = 'https://www.globalchange.gov/browse/indicators/us-heat-waves', 
                        target = '_blank'>
                        USGCRP</a></p>")
            )
          )
        ),
        tabItem(
          tabName = "interactive_map",
          h2("Interactive Map"),
          p("This interactive map contains 4 elements that provide a nice 
            overview of the overall narrative: People of Color Proportions, Heat 
            Sensitivity Index, Median Income, and the location of Cooling 
            Centers. These Cooling Centers open when temperatures of 95 degrees 
            Farenheit are reached, and in theory should be centered around the 
            most needy areas. Currently, they are mostly located around downtown 
            DC, the most densely populated areas, and those with the least green 
            spaces which makes sense on the surface. However, this does not 
            correlate with the most sensitive areas, which tend to be those with 
            higher proportions of People of Color.  This is an unnatural pattern 
            as nothing about ethnicitiy should lead to higher heat sensitivity, 
            so something else is at work. The question we explore for 
            the rest of the site is why this correlation is so high (0.88) and 
            what can be done about it. (One would be to redistribute the Cooling
            Centers from the Northwest (low HSI, high Income) to the Southeast 
            (high HSI, low Income)!"),
          br(),
          p("Play around with the map and look for relationships! The buttons on
            the side allow you to zoom in and out as well as select which layers
            you want to display. Notice how when you zoom in and out the markers
            change locations; these represent clusters of Cooling Centers that 
            become individual points as you zoom in allowing for a macro view of
            the general areas as well as their precise location."),
          tmapOutput(outputId = "interactive_map")
        ),
        
        tabItem(
          tabName = 'maps',
          h2('Map'),
          p("Using the dropdown menus, you can explore the relationship between 
            any two variables. The left menu contains the key variables of 
            interest while the right contains all of the available data. 
            Remember, we are trying to determine why the correlation between 
            People of Color prevalence and Heat Sensitivity is so high. I 
            recommend beginning with inputs to the HSI - income and health 
            factors. The chloropleths can provide great insight into spatial 
            relationships! 
            What do you notice?"),
          fluidRow(
            column(3,
             selectInput(
               inputId = 'var_select1',
               label = 'Plot 1',
               choices = 
                 c("Heat Sensitivity Index" = "hsi",
                   "Heat Exposure Index" = "hei",
                   "Median Household Income" = "median_household_income",
                   "Distance to Cooling Center" = "distance_to_cool" ,
                   "Asthma Prevalence" = "asthma",
                   "Obesity Prevalence" = "obesity",
                   "People of Color %" = "p_poc",
                   "Treecover" = "p_treecover")
             )
            ),
            column(3, offset = 3,
             selectInput(
               inputId = 'var_select2',
               label = 'Plot 2',
               choices = 
                 c("Heat Exposure Index" = "hei",
                   "Median Household Income" = "median_household_income",
                   "Distance to Cooling Center" = "distance_to_cool" ,
                   "Asthma Prevalence" = "asthma",
                   "Obesity Prevalence" = "obesity",
                   "People of Color %" = "p_poc",
                   "Treecover" = "p_treecover",
                   "Heart Disease Prevalence" = "chd",
                   "Disability Prevalence" = "p_disability",
                   "Heat Sensitivity Exposure Index" = "hsei",
                   "Unemployment Rate" = "unemployment_rate",
                   "% Working Outdoors" = "outdoor_work_p",
                   "% With Public Insurance" = "public_health_insurance_p",
                   "Population Density" = "pop_density",
                   "Total Population" = "totalpop",
                   "Impervious Surface %" = "p_impsurf",
                   "Avg. Airtemp" = "airtemp_mean")
             )
            )
          ),
          fluidRow(
            column(6, 
                   plotOutput(outputId = 'static_graph1')),
            column(6,
                   plotOutput(outputId = "static_graph2"))
          )
        ),
    
        tabItem(
          tabName = 'charts',
          h2('Plots'),
          p("Here we elucidate one of the key points: the high Heat Sensitivity 
            among People of Color is due to poor health and lower income, 
            not the Built Environment around them. Using the radio buttons you 
            can control which graphs you want to see. Notice how the Health 
            related plots all show strong positive correlations, while the Built
            Environment graphs have essentially no relationship, except for 
            income (a variable that cannot be ignored in a health conversation 
            as well!). None of this displays causality, but these relationships 
            are nevertheless eye-catching and paint a damning narrative: 
            minorities in Washington DC have similar access to heat relief in 
            physical spaces, but are disproportionately poor and sick, and thus 
            heat sensitive. Why? Past societal injustices have lead to 
            generationally compounding effects that have impoverished minority 
            groups - leading to poor health and few options to access quality 
            nutrition and care. To address these inequities, D.C. has done a 
            great job of providing relatively equal access to green space and 
            relief during acute instances of heat, but without addressing the 
            root cause of the problem. As a result, the problematic effects 
            remain."),
          radioButtons(inputId = "radio",
                       label = "View:",
                       choices = c("Health", "Built Environment")),
          fluidRow(
            column(5, offset = 1,
                   plotOutput(outputId = "plot1"),
                   br(),
                   br(),
                   plotOutput(outputId = "plot2")),
            column(5, offset = 0.25, 
                   plotOutput(outputId = "plot3"),
                   br(),
                   br(),
                   plotOutput(outputId = "plot4")),
            
          )
        ),
        tabItem(
          tabName = 'tables',
          h2('Summary table'),
          p("This summary table provides another view into health disparities
            from a race and wealth perspective - which are intimately linked in 
            DC. Census Tracts are grouped by Median Income buckets with means of
            the varibales being reported. We can clearly see that as income 
            rises, the proportion of People of Color falls, and nearly all 
            health related variables tend towards healthier values. Asthma and 
            obesity in particular are diseases that get exacerbated in heat as 
            the respiratory and circulatory system are known to be put under the 
            most stress. As a result, we see a furthering of the point that 
            People of Color are disadvantaged and are thus less able to adapt to 
            coming environmental challenges."),
            tableOutput(outputId = 'summary_table')
        ),
        tabItem(
          tabName = 'takeaway',
          h2('What can be done?'),
          HTML('<p>As the world gets hotter, those who have been marginalized in 
               the past and have dealt with systemic injustice are now ill 
               equipped for climactic change. Being the US capital, the 
               government is uniquely positioned to impact this community, and 
               set a tone for local urban governments across the country. By 
               being intentional with government funding, real change can be 
               created to positively impact the communities most at risk, and 
               truly ease their burden, as opposed to pursuing superficial 
               "green space" funding projects, while neglecting the root of the 
               problem. Some pathways to consider are:
               <ul>
                <li>Shift focus away from funding more public spaces and 
                    towards direct health interventions. (Do not abandon these 
                    efforts but shift relative focus)</li>
                <li>Redistribute Cooling Centers to more closely align with 
                    areas of highest Heat Sensitivity and away from less needy 
                    areas (Northwest -> Southeast)</li>
                <li>Provide high quality nutrition in schools</li>
                <li>Educate and inform on the relationship between asthma and 
                    heat stress</li>
                <li>Provide better access to respiratory healthcare options</li>
                <li>Improve educational quality in Southeast DC</li>
                <li>Improve access to higher paying job opportunities</li>
                <li>Provide financial incentives and community programs to 
                    promote household stability and home ownership</li>
               </ul>'
               )
        )
      )
    )
  )
# server ----------------------------------------

server <-
  function(input, output) { 
    
    
    # Reactive Objects --------------------------------------------------------
    
    static_graph1 <- 
      reactive({
        if (str_length(input$var_select1) <= 4){
          legend_title <- str_to_upper(input$var_select1)
        }
        else if (length(str_split(input$var_select1, "_")[[1]]) == 1){
          legend_title <- 
            paste0(
              str_to_title(input$var_select1), "%")
        }
        else if (input$var_select1 == "median_household_income"){
          legend_title <- "Median Income"
        }
        else if (input$var_select1 == "distance_to_cool"){
          legend_tile <- "Dist. to CC (Km)"
        }
        else{
          legend_title <- 
            paste0(
              str_to_title(
                str_replace(
                  str_remove(input$var_select1, "_"), 
                  "p","")
              ), "%")
        }
        if (input$var_select1 == "median_household_income"){
          g <- 
            ggplot(data = df) + 
            geom_sf(aes(fill = median_household_income)) + 
            theme_void()+
            scale_fill_continuous(name = "Median Income", 
                                  type = "viridis",
                                  breaks = 
                                    c(50000, 100000, 150000, 200000, 250000),
                                  labels = 
                                    c('50K', '100K', "150K", "200K", "250K"))+
            theme(legend.position = "left",
                  legend.title = element_text(size = 18),
                  legend.text = element_text(size = 14))
        }
        else{
          g <-
            ggplot(data = df) + 
            geom_sf(aes(fill = .data[[input$var_select1]])) + 
            theme_void()+
            scale_fill_continuous(name = legend_title, type = "viridis")+
            theme(legend.position = "left",
                  legend.title = element_text(size = 18),
                  legend.text = element_text(size = 14))
        }
        g
      })
    
    static_graph2 <- 
      reactive({
        if (str_length(input$var_select2) <= 4){
          legend_title <- str_to_upper(input$var_select2)
        }
        else if (input$var_select2 == "median_household_income"){
          legend_title <- "Median Income"
        }
        else if (input$var_select2 == "distance_to_cool"){
          legend_title <- "Dist. to CC (Km)"
        }
        else if (input$var_select2 == "pop_density"){
          legend_title <- "Pop Density"
        }
        else if (input$var_select2 == "totalpop"){
          legend_title <- "Total Pop"
        }
        else if (input$var_select2 == "airtemp_mean"){
          legend_title <- "Avg. Airtemp"
        }
        else if (input$var_select2 == "p_impsurf"){
          legend_title <- "Imp. Surface%"
        }
        else if (length(str_split(input$var_select2, "_")[[1]]) == 1){
          legend_title <- 
            paste0(
              str_to_title(input$var_select2), "%")
        }
        else if (str_detect(input$var_select2, "p_|_p")){
          legend_title <- 
            paste0(
              str_trim(
                str_to_title(
                  str_replace_all(input$var_select2, "_p|p_|_", " ")
                ), "both"),
              "%")
          
        }
        else{
          legend_title <- str_to_title(str_replace(input$var_select2, "_", " "))
        }
        if (input$var_select2 == "median_household_income"){
          g <- 
            ggplot(data = df) + 
            geom_sf(aes(fill = median_household_income)) + 
            theme_void()+
            scale_fill_continuous(name = "Median Income", 
                                  type = "viridis",
                                  breaks = 
                                    c(50000, 100000, 150000, 200000, 250000),
                                  labels = 
                                    c('50K', '100K', "150K", "200K", "250K"))+
            theme(legend.position = "right",
                  legend.title = element_text(size = 18),
                  legend.text = element_text(size = 14))
        }
        else{
          g <-
            ggplot(data = df) + 
            geom_sf(aes(fill = .data[[input$var_select2]])) + 
            theme_void()+
            scale_fill_continuous(name = legend_title, type = "viridis")+
            theme(legend.position = "right",
                  legend.title = element_text(size = 18),
                  legend.text = element_text(size = 14))
        }
        g
      })
    
    plot_1 <- 
      reactive({
        if (input$radio == "Health"){
          g <- 
            static_plot1a
        }
        else {
          g <-
            static_plot1b
        }
        g
      })
    
    plot_2 <- 
      reactive({
        if (input$radio == "Health"){
          g <-
            static_plot2a
        }
        else {
          g <-
            static_plot2b
        }
        g
      })
    
    plot_3 <- 
      reactive({
        if (input$radio == "Health"){
          g <-
            static_plot3a
        }
        else {
          g <-
            static_plot3b
        }
        g
      })
    
    plot_4 <- 
      reactive({
        if (input$radio == "Health"){
          g <-
           static_plot4a
        }
        else {
          g <-
            static_plot4b
        }
        g
      })
    
    
    
    # Outputs -----------------------------------------------------------------
    
    output$interactive_map <- 
      renderTmap(
       interactive_map
      )
    
    output$static_graph1 <- 
      renderPlot(
        static_graph1()
      )
    
    output$static_graph2 <- 
      renderPlot(
        static_graph2()
      )
    
    output$summary_table <- 
      renderTable(
        data_table, 
        align = 'c')
    
    output$plot1 <- 
      renderPlot(
        plot_1()
      )
    
    output$plot2<- 
      renderPlot(
        plot_2()
      )
    
    output$plot3 <- 
      renderPlot(
        plot_3()
      )
    
    output$plot4 <- 
      renderPlot(
        plot_4()
      )
  }

# knit and run app ------------------------------

shinyApp(ui, server)
