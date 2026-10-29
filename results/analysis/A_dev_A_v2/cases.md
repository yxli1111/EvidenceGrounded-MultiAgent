# A: M2/M3 case list (PRELIMINARY pilot)

## M2 correct->wrong (4)

- **5a7cc50e554299452d57ba3e** (comparison) Q: Which band, Letters to Cleo or Screaming Trees, had more members?
  - gold: `Letters to Cleo` | draft: `Letters to Cleo` | final: `Screaming Trees`
  - reason: content: answer replaced by a different answer
  - feedback: The short answer is incorrect because it states 'Letters to Cleo' as having more members, but Screaming Trees had four original members and later added a fifth member, making them equal in number to Letters to Cleo, whic | fix: no
  - feedback: The explanation incorrectly concludes that Letters to Cleo had more members than Screaming Trees without considering that Screaming Trees had four original members and later added a fifth member, which means they had a t | fix: Screaming Trees originally had four members: Mark Lanegan, Gary Lee Conner, Van Conner, and Mark Pickerel, with Barrett Martin replacing Pickerel later,
- **5a8ef2a75542995a26add583** (bridge) Q: Seven Brief Lessons on Physics was written by an Italian physicist that has worked in France since what year?
  - gold: `2000` | draft: `2000` | final: `since 2000`
  - reason: format: final contains gold but is longer (sentence / extra words)
  - feedback: The short answer '2000' is correct as it states the year Carlo Rovelli began working in France, but it does not explicitly answer the question about the year he has worked in France since, which is implied to be 2000. Th | fix: since 2000
- **5abbf698554299114383a0b5** (bridge) Q: Which British first-generation jet-powered medium bomber was used in the South West Pacific theatre of World War II?
  - gold: `English Electric Canberra` | draft: `English Electric Canberra` | final: `Savoia-Marchetti SM.79`
  - reason: content: answer replaced by a different answer
  - feedback: The English Electric Canberra was not used during World War II; it was introduced in 1951, after the war had ended.
- **5ae7a8175542993210983ed8** (bridge) Q: Which other Mexican Formula One race car driver has held the podium besides the Force India driver born in 1990?
  - gold: `Pedro Rodríguez` | draft: `Pedro Rodríguez` | final: `Sergio Pérez`
  - reason: content: answer replaced by a different answer
  - feedback: The short answer incorrectly identifies Pedro Rodríguez as the only other Mexican driver to have held the podium, while the context states that Sergio Pérez is the only other Mexican to finish on the podium. | fix: Sergio Pérez

## M2 error (1)

- **5a79311755429970f5fffe67** (bridge) Q: A Japanese manga series based on a 16 year old high school student Ichitaka Seto, is written and illustrated by someone born in what year?
  - gold: `1962` | draft: `1962` | final: ``
  - error: A.M2.self_revise: unparseable LLM output after 2 attempt(s): Expecting ',' delimiter: line 1 column 111 (char 110)

## M2 wrong->correct (5)

- **5a7bbb64554299042af8f7cc** (bridge) Q: Who is older, Annie Morton or Terry Richardson?
  - gold: `Terry Richardson` | draft: `Annie Morton` | final: `Terry Richardson`
  - feedback: The short answer is incorrect because it states that Annie Morton is older, while the context clearly indicates that Terry Richardson is older. | fix: Terry Richardson
  - feedback: The explanation incorrectly states that Terry Richardson is older than Annie Morton, which contradicts the short answer. It also fails to provide the correct age difference in a clear manner. | fix: Annie Morton was born on October 8, 1970, while Terry Richardson was born on August 14, 1965. This makes Terry Richardson older than Annie Morton by app
- **5a80721b554299485f5985ef** (bridge) Q: The Livesey Hal War Memorial commemorates the fallen of which war, that had over 60 million casualties?
  - gold: `World War II` | draft: `World War I and World War II` | final: `World War II`
  - feedback: The short answer includes both World War I and World War II, but the question specifically asks for the war that had over 60 million casualties, which is only World War II. | fix: World War II
- **5a8e3ea95542995a26add48d** (bridge) Q: The director of the romantic comedy "Big Stone Gap" is based in what New York city?
  - gold: `Greenwich Village, New York City` | draft: `Greenwich Village` | final: `Greenwich Village, New York City`
  - feedback: The short answer is correct as it identifies the city where Adriana Trigiani is based, but it does not directly answer the question about the director of 'Big Stone Gap' being based in a New York city. | fix: Greenwich Village, New York City
- **5ab56e32554299637185c594** (comparison) Q: Are Random House Tower and 888 7th Avenue both used for real estate?
  - gold: `no` | draft: `yes` | final: `no`
  - feedback: The short answer 'yes' implies that both buildings are used for real estate, but it does not clarify that 888 7th Avenue is primarily an office building and not a residential real estate property. | fix: no
  - feedback: The explanation states that 888 7th Avenue carries the Vornado Realty Trust corporate headquarters, but it does not clarify that this is a commercial use, which is different from residential real estate. | fix: 888 7th Avenue is primarily an office building.
- **5abf63f15542997ec76fd3ea** (bridge) Q: Alexander Kerensky was defeated and destroyed by the Bolsheviks in the course of a civil war that ended when ?
  - gold: `October 1922` | draft: `1923` | final: `October 1922`
  - feedback: The short answer '1923' is incorrect because the Russian Civil War officially ended in October 1922, not 1923, although Red control was assured in 1923. | fix: October 1922

## M3 correct->wrong (5)

- **5a760ab65542994ccc918697** (bridge) Q: Alfred Balk served as the secretary of the Committee on the Employment of Minority Groups in the News Media under which United States Vice President?
  - gold: `Nelson Rockefeller` | draft: `Nelson Rockefeller` | final: `no`
  - reason: content: answer replaced by a different answer
  - feedback: The short answer is incorrect because Alfred Balk served under New York Governor Nelson Rockefeller, not under a United States Vice President as the question asks.
- **5a7cc50e554299452d57ba3e** (comparison) Q: Which band, Letters to Cleo or Screaming Trees, had more members?
  - gold: `Letters to Cleo` | draft: `Letters to Cleo` | final: `Screaming Trees`
  - reason: content: answer replaced by a different answer
  - feedback: The short answer is incorrect because it states 'Letters to Cleo' as having more members, but the explanation shows that Screaming Trees originally had four members, while Letters to Cleo had six members, which means Let | fix: Screaming Trees
  - feedback: The explanation correctly states the number of members in both bands, but it fails to clarify that Screaming Trees had four original members and later had five members when including Barrett Martin, which is relevant to  | fix: Screaming Trees originally had four members and later had five members with Barrett Martin.
- **5ab51dae5542991779162d82** (comparison) Q: Are both Elko Regional Airport and Gerald R. Ford International Airport located in Michigan?
  - gold: `no` | draft: `no` | final: `yes`
  - reason: content: answer replaced by a different answer
  - feedback: The short answer 'no' is incorrect because Gerald R. Ford International Airport is located in Michigan, while Elko Regional Airport is not, which means the answer should reflect that one of the two airports is indeed in  | fix: yes
- **5abbf698554299114383a0b5** (bridge) Q: Which British first-generation jet-powered medium bomber was used in the South West Pacific theatre of World War II?
  - gold: `English Electric Canberra` | draft: `English Electric Canberra` | final: ``
  - reason: content: answer replaced by a different answer
  - feedback: The short answer is incorrect because the English Electric Canberra was not used during World War II; it was introduced in 1951, after the war had ended.
- **5ae7a8175542993210983ed8** (bridge) Q: Which other Mexican Formula One race car driver has held the podium besides the Force India driver born in 1990?
  - gold: `Pedro Rodríguez` | draft: `Pedro Rodríguez` | final: `Sergio Pérez`
  - reason: content: answer replaced by a different answer
  - feedback: The short answer is incorrect because it does not address the question about which other Mexican driver has held the podium besides Sergio Pérez, who is the Force India driver born in 1990. | fix: Sergio Pérez
  - feedback: The explanation incorrectly states that Sergio Pérez is the only other Mexican to finish on the podium, which contradicts the context that indicates he is the only other Mexican to finish on the podium besides Pedro Rodr | fix: Sergio Pérez is the only other Mexican to finish on the podium besides Pedro Rodríguez.

## M3 wrong->correct (6)

- **5a7bbb64554299042af8f7cc** (bridge) Q: Who is older, Annie Morton or Terry Richardson?
  - gold: `Terry Richardson` | draft: `Annie Morton` | final: `Terry Richardson`
  - feedback: The short answer is incorrect; it states Annie Morton is older, but Terry Richardson is actually older. | fix: Terry Richardson
  - feedback: The explanation incorrectly states that Terry Richardson is older than Annie Morton by approximately 5 years; the actual age difference is about 5 years and 2 months. | fix: This makes Terry Richardson older than Annie Morton by approximately 5 years and 2 months.
- **5a80721b554299485f5985ef** (bridge) Q: The Livesey Hal War Memorial commemorates the fallen of which war, that had over 60 million casualties?
  - gold: `World War II` | draft: `World War I and World War II` | final: `World War II`
  - feedback: The short answer includes both World War I and World War II, but the question specifically asks for the war that had over 60 million casualties, which is only World War II. | fix: World War II
- **5ab56e32554299637185c594** (comparison) Q: Are Random House Tower and 888 7th Avenue both used for real estate?
  - gold: `no` | draft: `yes` | final: `no`
  - feedback: The short answer 'yes' implies that both buildings are used for real estate, but it does not clarify that 888 7th Avenue is primarily an office building and not a residential real estate property. | fix: no
- **5ac23ff0554299636651994d** (bridge) Q: When was Poison's album "Shut Up, Make Love" released?
  - gold: `2000` | draft: `March 14, 2000` | final: `2000`
  - feedback: The short answer incorrectly states the release date of the album instead of the single, which is what the question asks for. | fix: 2000
  - feedback: The explanation incorrectly concludes that the release date of the album is the same as the release date of the single, which is not accurate as the question specifically asks for the single's release date. | fix: The song 'Shut Up, Make Love' was released in 2000 as the first single from the album 'Crack a Smile...and More!'. The album was released on March 14, 2
- **5ae22b8d554299234fd0440f** (bridge) Q: What was the father of Kasper Schmeichel voted to be by the IFFHS in 1992?
  - gold: `World's Best Goalkeeper` | draft: `IFFHS World's Best Goalkeeper` | final: `World's Best Goalkeeper`
  - feedback: The short answer does not directly answer the question about what Peter Schmeichel was voted to be by the IFFHS in 1992; it states the name of the award instead of the specific title or recognition he received. | fix: World's Best Goalkeeper
  - feedback: The explanation states that the award recognizes the most outstanding goalkeeper of the year, which is true, but it does not clarify that Peter Schmeichel was specifically voted as the World's Best Goalkeeper for the yea | fix: Peter Schmeichel was voted the IFFHS World's Best Goalkeeper for the year 1992.
- **5ae4a3265542995ad6573de5** (comparison) Q: Hayden is a singer-songwriter from Canada, but where does Buck-Tick hail from?
  - gold: `Fujioka, Gunma` | draft: `Japan` | final: `Fujioka, Gunma`
  - feedback: The short answer is incorrect; Buck-Tick hails from Japan, but the answer should specify the city or region, which is Fujioka, Gunma. | fix: Fujioka, Gunma
