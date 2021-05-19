PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX wd:   <http://www.wikidata.org/entity/>
PREFIX wdt:  <http://www.wikidata.org/prop/direct/>
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX : <urn:absolute:/problem1a#>

INSERT {
    ?person rdfs:type :Person ;
              :personName ?personLabel ;
      		  :alumnusOf  ?univLabel ;
    		  :employeeOf ?organisation;
    		  :bornIn ?birthPlace .
    ?birthPlace rdfs:type :Place ;
            :countryName ?countryLabel ;
    		:placeName ?placeLabel .
    ?hq rdfs:type :Place ;
            :countryName ?country2Label ;
    		:placeName ?hqLabel .
    ?organisation rdfs:type :Organisation ;
             :organisationName ?orgcountryLabel ;
    		 :placeName ?hq . #locatedIn
}
WHERE {
  SERVICE <https://query.wikidata.org/sparql>  {
      ?person wdt:P31 wd:Q5; #human
        wdt:P69 ?univ; #educated at
        rdfs:label ?personLabel; 
		wdt:P108 ?organisation; #employer #optional
        wdt:P19 ?birthPlace. #place of birth #optional
            
      ?univ wdt:P31 wd:Q3918; #instance of -university
        wdt:P17 wd:Q34; #country of -Sweden
        wdt:P571 ?founded; #inception
        rdfs:label ?univLabel.
	
      ?birthPlace wdt:P131 ?country. #located in administrative part of entitity
      ?country wdt:P31 wd:Q6256. #instance of -country
        
      ?organisation wdt:P159 ?hq .
      ?hq wdt:P131 ?country2 .
        
      ?country2 rdfs:label ?country2Label .
      ?hq rdfs:label ?hqLabel .
       
      ?birthPlace rdfs:label ?placeLabel. 
      ?country rdfs:label ?countryLabel.  
      ?organisation rdfs:label ?organisationLabel.
        
      FILTER(LANGMATCHES(LANG(?personLabel), "en"))
      FILTER(LANGMATCHES(LANG(?univLabel), "en"))
      FILTER(LANGMATCHES(LANG(?organisationLabel), "en"))
      FILTER(LANGMATCHES(LANG(?placeLabel), "en"))
      FILTER(LANGMATCHES(LANG(?countryLabel), "en"))
      FILTER(LANGMATCHES(LANG(?hqLabel), "en"))
      FILTER(LANGMATCHES(LANG(?country2Label), "en"))
	}
}
