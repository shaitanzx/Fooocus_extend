##import modules.scripts as scripts
import gradio as gr
import os
import platform
import subprocess as sp


##from modules import images
##from modules.processing import process_images, Processed
##from modules.processing import Processed
##from modules.shared import opts, cmd_opts, state


from ..build_dynamic_prompt import *
from ..main import *
from ..model_lists import *
from ..csv_reader import *

from ..one_button_presets import OneButtonPresets
OBPresets = OneButtonPresets()

basemodelslist = ["SD1.5", "SDXL", "Stable Cascade", "Anime Model"]
#subjects = ["all","object","animal","humanoid", "landscape", "concept"]
subjects =["------ all"]
subjectsubtypesobject = ["all"]
subjectsubtypeshumanoid = ["all"]
subjectsubtypesconcept = ["all"]
artists = ["all", "all (wild)", "none", "popular", "greg mode", "3D",	"abstract",	"angular", "anime"	,"architecture",	"art nouveau",	"art deco",	"baroque",	"bauhaus", 	"cartoon",	"character",	"children's illustration", 	"cityscape", "cinema",	"clean",	"cloudscape",	"collage",	"colorful",	"comics",	"cubism",	"dark",	"detailed", 	"digital",	"expressionism",	"fantasy",	"fashion",	"fauvism",	"figurativism",	"gore",	"graffiti",	"graphic design",	"high contrast",	"horror",	"impressionism",	"installation",	"landscape",	"light",	"line drawing",	"low contrast",	"luminism",	"magical realism",	"manga",	"melanin",	"messy",	"monochromatic",	"nature",	"nudity",	"photography",	"pop art",	"portrait",	"primitivism",	"psychedelic",	"realism",	"renaissance",	"romanticism",	"scene",	"sci-fi",	"sculpture",	"seascape",	"space",	"stained glass",	"still life",	"storybook realism",	"street art",	"streetscape",	"surrealism",	"symbolism",	"textile",	"ukiyo-e",	"vibrant",	"watercolor",	"whimsical"]
imagetypes = ["all", "all - force multiple", "all - anime",  "none", "photograph", "octane render","digital art","concept art", "painting", "portrait", "anime", "only other types", "only templates mode", "dynamic templates mode", "art blaster mode", "quality vomit mode", "color cannon mode", "unique art mode", "massive madness mode", "photo fantasy mode", "subject only mode", "fixed styles mode", "the tokinator"]
promptmode = ["at the back", "in the front"]
promptcompounder = ["1", "2", "3", "4", "5"]
ANDtogglemode = ["none", "automatic", "prefix AND prompt + suffix", "prefix + prefix + prompt + suffix"]
seperatorlist = ["comma", "AND", "BREAK"]
genders = ["all", "male", "female"]

prompt_enhancers = ["none", "superprompt-v1"]

qualitymodelist = ["highest", "gated"]
qualitykeeplist = ["keep used","keep all"]

amountofflufflist = ["none", "dynamic", "short", "medium", "long"]

#for autorun and upscale
sizelist = ['all','current ratio','ultraheight','portrait','square','wide','ultrawide','random ratio']
basesizelist = ["512", "768", "1024"]

modellist = ['all','current model','random model']
#modellist.insert(0,"all")
#modellist.insert(0,"currently selected model") # First value us the currently selected model

upscalerlist = get_upscalers()
upscalerlist.insert(0,"automatic")
upscalerlist.insert(0,"all")

samplerlist = get_samplers()
samplerlist.insert(0,"all")

#for img2img
img2imgupscalerlist = get_upscalers_for_img2img()
img2imgupscalerlist.insert(0,"automatic")
img2imgupscalerlist.insert(0,"all")

img2imgsamplerlist = get_samplers_for_img2img()
img2imgsamplerlist.insert(0,"all")

#for ultimate SD upscale

seams_fix_types = ["None","Band pass","Half tile offset pass","Half tile offset pass + intersections"]
redraw_modes = ["Linear","Chess","None"]

#folder stuff
folder_symbol = '\U0001f4c2'  # 📂
sys.path.append(os.path.abspath(".."))

# Load up stuff for personal artists list, if any
# find all artist files starting with personal_artits in userfiles
script_dir = os.path.dirname(os.path.abspath(__file__))  # Script directory
userfilesfolder = os.path.join(script_dir, "../userfiles/" )
for filename in os.listdir(userfilesfolder):
    if(filename.endswith(".csv") and filename.startswith("personal_artists") and filename != "personal_artists_sample.csv"):
        name = os.path.splitext(filename)[0]
        name = name.replace("_"," ",-1).lower()
        # directly insert into the artists list
        artists.insert(2, name)

# on startup, check if we have a config file, or else create it
config = load_config_csv()

# load subjects stuff from config
generatevehicle = True
generateobject = True
generatefood = True
generatebuilding = True
generatespace = True
generateflora = True

generateanimal = True
generatebird = True
generatecat = True
generatedog = True
generateinsect = True
generatepokemon = True
generatemarinelife = True

generatemanwoman = True
generatemanwomanrelation = True
generatemanwomanmultiple = True
generatefictionalcharacter = True
generatenonfictionalcharacter = True
generatehumanoids = True
generatejob = True
generatefirstnames = True

generatelandscape = True
generatelocation = True
generatelocationfantasy = True
generatelocationscifi = True
generatelocationvideogame = True
generatelocationbiome = True
generatelocationcity = True

generateevent = True
generateconcepts = True
generatepoemline = True
generatesongline = True
generatecardname = True
generateepisodetitle = True
generateconceptmixer = True


for item in config:
        # objects
        if item[0] == 'subject_vehicle' and item[1] != 'on':
            generatevehicle = False
        if item[0] == 'subject_object' and item[1] != 'on':
            generateobject = False
        if item[0] == 'subject_food' and item[1] != 'on':
            generatefood = False
        if item[0] == 'subject_building' and item[1] != 'on':
            generatebuilding = False
        if item[0] == 'subject_space' and item[1] != 'on':
            generatespace = False
        if item[0] == 'subject_flora' and item[1] != 'on':
            generateflora = False
        # animals
        if item[0] == 'subject_animal' and item[1] != 'on':
            generateanimal = False
        if item[0] == 'subject_bird' and item[1] != 'on':
            generatebird = False
        if item[0] == 'subject_cat' and item[1] != 'on':
            generatecat = False
        if item[0] == 'subject_dog' and item[1] != 'on':
            generatedog = False
        if item[0] == 'subject_insect' and item[1] != 'on':
            generateinsect = False
        if item[0] == 'subject_pokemon' and item[1] != 'on':
            generatepokemon = False
        if item[0] == 'subject_marinelife' and item[1] != 'on':
            generatemarinelife = False
        # humanoids
        if item[0] == 'subject_manwoman' and item[1] != 'on':
            generatemanwoman = False
        if item[0] == 'subject_manwomanrelation' and item[1] != 'on':
            generatemanwomanrelation = False
        if item[0] == 'subject_manwomanmultiple' and item[1] != 'on':
            generatemanwomanmultiple = False
        if item[0] == 'subject_fictional' and item[1] != 'on':
            generatefictionalcharacter = False
        if item[0] == 'subject_nonfictional' and item[1] != 'on':
            generatenonfictionalcharacter = False
        if item[0] == 'subject_humanoid' and item[1] != 'on':
            generatehumanoids = False
        if item[0] == 'subject_job' and item[1] != 'on':
            generatejob = False
        if item[0] == 'subject_firstnames' and item[1] != 'on':
            generatefirstnames = False
        # landscape
        if item[0] == 'subject_location' and item[1] != 'on':
            generatelocation = False
        if item[0] == 'subject_location_fantasy' and item[1] != 'on':
            generatelocationfantasy = False
        if item[0] == 'subject_location_scifi' and item[1] != 'on':
            generatelocationscifi = False
        if item[0] == 'subject_location_videogame' and item[1] != 'on':
            generatelocationvideogame = False
        if item[0] == 'subject_location_biome' and item[1] != 'on':
            generatelocationbiome = False
        if item[0] == 'subject_location_city' and item[1] != 'on':
            generatelocationcity = False
        # concept
        if item[0] == 'subject_event' and item[1] != 'on':
            generateevent = False
        if item[0] == 'subject_concept' and item[1] != 'on':
            generateconcepts = False
        if item[0] == 'subject_poemline' and item[1] != 'on':
            generatepoemline = False
        if item[0] == 'subject_songline' and item[1] != 'on':
            generatesongline = False
        if item[0] == 'subject_cardname' and item[1] != 'on':
            generatecardname = False
        if item[0] == 'subject_episodetitle' and item[1] != 'on':
            generateepisodetitle = False
        if item[0] == 'subject_conceptmixer' and item[1] != 'on':
            generateconceptmixer = False

# build up all subjects we can choose based on the loaded config file
if(generatevehicle or generateobject or generatefood or generatebuilding or generatespace or generateflora):
    subjects.append("--- object - all")
    if(generateobject):
          subjects.append("object - generic")
    if(generatevehicle):
          subjects.append("object - vehicle")
    if(generatefood):
          subjects.append("object - food")
    if(generatebuilding):
          subjects.append("object - building")
    if(generatespace):
          subjects.append("object - space")
    if(generateflora):
          subjects.append("object - flora")
          
if(generateanimal or generatebird or generatecat or generatedog or generateinsect or generatepokemon or generatemarinelife):
    subjects.append("--- animal - all")
    if(generateanimal):
        subjects.append("animal - generic")
    if(generatebird):
        subjects.append("animal - bird")
    if(generatecat):
        subjects.append("animal - cat")
    if(generatedog):
        subjects.append("animal - dog")
    if(generateinsect):
        subjects.append("animal - insect")
    if(generatemarinelife):
        subjects.append("animal - marine life")
    if(generatepokemon):
        subjects.append("animal - pokémon")

if(generatemanwoman or generatemanwomanrelation or generatefictionalcharacter or generatenonfictionalcharacter or generatehumanoids or generatejob or generatemanwomanmultiple):
    subjects.append("--- human - all")
    if(generatemanwoman):
        subjects.append("human - generic")
    if(generatemanwomanrelation):
        subjects.append("human - relations")
    if(generatenonfictionalcharacter):
        subjects.append("human - celebrity")
    if(generatefictionalcharacter):
        subjects.append("human - fictional")
    if(generatehumanoids):
        subjects.append("human - humanoids")
    if(generatejob):
        subjects.append("human - job/title")
    if(generatefirstnames):
        subjects.append("human - first name")
    if(generatemanwomanmultiple):
        subjects.append("human - multiple")

if(generatelandscape or generatelocation or generatelocationfantasy or generatelocationscifi or generatelocationvideogame or generatelocationbiome or generatelocationcity):
    subjects.append("--- landscape - all")
    if(generatelocation):
        subjects.append("landscape - generic")
    if(generatelocationfantasy):
        subjects.append("landscape - fantasy")
    if(generatelocationscifi):
        subjects.append("landscape - sci-fi")
    if(generatelocationvideogame):
        subjects.append("landscape - videogame")
    if(generatelocationbiome):
        subjects.append("landscape - biome")
    if(generatelocationcity):
        subjects.append("landscape - city")

if(generateevent or generateconcepts or generatepoemline or generatesongline or generatecardname or generateepisodetitle or generateconceptmixer):
    subjects.append("--- concept - all")
    if(generateevent):
        subjects.append("concept - event")
    if(generateconcepts):
        subjects.append("concept - the x of y")
    if(generatepoemline):
        subjects.append("concept - poem lines")
    if(generatesongline):
        subjects.append("concept - song lines")
    if(generatecardname):
        subjects.append("concept - card names")
    if(generateepisodetitle):
        subjects.append("concept - episode titles")
    if(generateconceptmixer):
        subjects.append("concept - mixer")
         


# do the same for the subtype subjects
# subjectsubtypesobject = ["all"]
# subjectsubtypeshumanoid = ["all"]
# subjectsubtypesconcept = ["all"]

# objects first
if(generateobject):
     subjectsubtypesobject.append("generic objects")
if(generatevehicle):
     subjectsubtypesobject.append("vehicles")
if(generatefood):
     subjectsubtypesobject.append("food")
if(generatebuilding):
     subjectsubtypesobject.append("buildings")
if(generatespace):
     subjectsubtypesobject.append("space")
if(generateflora):
     subjectsubtypesobject.append("flora")

# humanoids (should I review descriptions??)
if(generatemanwoman):
     subjectsubtypeshumanoid.append("generic humans")
if(generatemanwomanrelation):
     subjectsubtypeshumanoid.append("generic human relations")
if(generatenonfictionalcharacter):
     subjectsubtypeshumanoid.append("celebrities e.a.")
if(generatefictionalcharacter):
     subjectsubtypeshumanoid.append("fictional characters")
if(generatehumanoids):
     subjectsubtypeshumanoid.append("humanoids")
if(generatejob):
     subjectsubtypeshumanoid.append("based on job or title")
if(generatefirstnames):
     subjectsubtypeshumanoid.append("based on first name")
if(generatemanwomanmultiple):
     subjectsubtypeshumanoid.append("multiple humans")


# concepts
if(generateevent):
     subjectsubtypesconcept.append("event")
if(generateconcepts):
     subjectsubtypesconcept.append("the X of Y concepts")
if(generatepoemline):
     subjectsubtypesconcept.append("lines from poems")
if(generatesongline):
     subjectsubtypesconcept.append("lines from songs")
if(generatecardname):
     subjectsubtypesconcept.append("names from card based games")
if(generateepisodetitle):
     subjectsubtypesconcept.append("episode titles from tv shows")
if(generateconceptmixer):
     subjectsubtypesconcept.append("concept mixer")

main_mark1="""
                            <font size="2">
                            
                            Set the One Button Preset to __"Custom..."__ to show all settings. The settings will give you more control over what you wish to generate.

                            Choose __"All (random)..."__ to get a random preset each prompt generation.
                            </font>
                            """
main_mark2="""
                            ### Description
                            
                            <font size="2">
                            For generate use copy prompt1-prompt5 from Workflow Tab to prompt box

                            This generator will generate a complete full prompt for you and generate the image, based on randomness. You can increase the slider, to include more things to put into the prompt. 
                            Recommended is keeping it around 3-7. Use 10 at your own risk.

                            There are a lot of special things build in, based on various research papers. Just try it, and let it surprise you.

                            Add additional prompting to the prefix, suffix in this screen. The actual prompt fields are ignored. Negative prompt is in the respective tab.
                            </font>
                            
                            ### Subject Types
                            
                            <font size="2">
                            You can choose a certain subject type. Choose the all version to randomly choose between the subtypes. Iff you want to generate something more specific, choose the subtype. It has the following types:  
                            
                            1. object - Can be a random object, a building, vehicle, space or flora.
                            
                            2. animal - A random (fictional) animal. Has a chance to have human characteristics, such as clothing added.  
                            
                            3. humanoid - A random humanoid, males, females, fantasy types, fictional and non-fictional characters. Can add clothing, features and a bunch of other things.  
                            
                            4. landscape - A landscape, choose a cool location.
                            
                            5. concept - Can be a concept, such as "a X of Y", or an historical event such as "The Trojan War". It can also generate a line from a poem or a song. 

                            After choosing object, humanoid or concept a subselection menu will show. You can pick further details here. When choosing humanoid, you can also select the gender you wish to generate.

                           
                            ### gender (only available for human generations):

                            1. all - selects randomly

                            2. male

                            3. female

                            </font>
                            
                            ### Artists
                            
                            <font size="2">
                            Artists have a major impact on the result.
                            
                            1. all - it will cohesivly add about between 0-3 artists and style description. 
                            
                            2. all (wild) - it will randomly select between 0-3 artists out of 3483 artists for your prompt.

                            3. greg mode - Will add greg, or many other popular artists into your prompt. Will also add a lot of quality statements. 

                            Others will select within that artist category
                            
                            You can turn it off and maybe add your own in the prefix or suffix prompt fields
                            </font>

                            ### type of image

                            <font size="2">
                            There are an immense number of image types, not only paintings and photo's, but also isometric renders and funko pops.
                            You can however, overwrite it with the most popular ones.


                            1. all --> normally picks a image type as random. Can choose a 'other' more unique type.

                            2. all - force multiple  --> idea by redditor WestWordHoeDown, it forces to choose between 2 and 3 image types

                            3. all - anime --> Chooses from anime styles, for support of Anime Model mode.

                            4. none --> Turns off image type generation
                            
                            5. photograph

                            6. octane render

                            7. digital art

                            8. concept art

                            9. painting

                            10. portrait

                            11. anime
                            
                            12. only other types --> Will pick only from the more unique types, such as stained glass window or a funko pop

                            All modes below are considered a special image type mode.

                            13. only templates mode --> Will only choose from a set of wildcarded prompt templates. Templates have been gathered from various sources, such as CivitAI, prompthero, promptbook, etc.

                            only templates mode is perfect for beginners, who want to see some good results fast.

                            14. dynamic templates mode --> A method that uses prompts in a more natural language.

                            15. art blaster mode --> Special generation mode that focusses on art movements, stylings and artists.

                            16. quality vomit mode --> Special generation mode that focusses on qualifiers and stylings.

                            17. color cannon mode --> Special generation mode that focusses on color scheme's and moods.

                            18. unique art mode --> Special generation mode that focusses on other image types, art movements, stylings and lighting.

                            19. massive madness mode --> Special generation mode, creates prompt soup. Almost pure randomness.

                            20. photo fantasy mode --> Special generation mode that focusses on photographs, cameras, lenses and lighting.

                            21. subject only mode --> Will only generate a subject, with no additional frills.

                            22. fixed styles mode --> Generate a subject on top of a fixed style.

                            23. the tokinator --> Complete random word gibberish mode, use at own risk

                            ### One in X chance to use special image type mode

                            <font size="2">
                            This controls how often it will pick a special generation mode. It is a 1 in X chance. So lower means more often. This will only be applied of "type of image" is set to "all" and there is no Overwrite type of image set.

                            When set to 1, it will always pick a random special generation mode. When set to 20, it is a 1 in 20 chance this will happen.
                            </font>
                            
                            ### Overwrite subject

                            When you fill in the Overwrite subject field, that subject will be used to build the dynamic prompt around. It is best, if you set the subject type to match the subject. For example, set it to humanoid if you place a person in the override subject field.
                            
                            This way, you can create unlimited variants of a subject.

                            ### Smart subject tries to determine what to and not to generate based on your subject. Example, if your Overwrite subject is formed like this: Obese man wearing a kimono
                            
                            It will then recognize the body type and not generate it. It also recognizes the keyword wearing, and will not generate an outfit.

                            ### Overwrite outfit

                            When you fill in the override outfit field, it will generate an outfit in the prompt based on the given value. It can be used in combination with override subject, but does not have to be. It works best with smaller descriptions of the outfit.

                            An example would be: space suit, red dress, cloak.

                            Works best when you set the subjects to to humanoid.
                            
                            ### Other prompt fields

                            The existing prompt and negative prompt fields are ignored.
                            
                            Add a prompt prefix, suffix in the respective fields. Add negative prompt in the negative prompt tab. They will be automatically added during processing.

                            These can be used to add textual inversion and LoRA's to always apply. They can also be used to add your models trigger words.

                            Please read the custom_files documentation on how to apply random textual inversion and LoRA's.

                            </font>

                            ### Filter values
                            <font size="2">
                            You can put comma seperated values here, those will be ignored from any list processing. For example, adding ""film grain, sepia"", will make these values not appear during generation.

                            For advanced users, you can create a permanent file in \\OneButtonPrompt\\userfiles\\ called antilist.csv
                            
                            This way, you don't ever have to add it manually again. This file won't be overwritten during upgrades.

                            Idea by redditor jonesaid.

                            </font>
                            """
wf_mark1="""
                     <font size="2"> 
                     Workflow assist, suggestions by redditor Woisek.

                     With Workflow mode, you turn off the automatic generation of new prompts on 'generate', and it will use the Workflow prompt field instead. So you can work and finetune any fun prompts without turning of the script.

                     You can use One Button Prompt wildcards in the workflow prompt. For example -outfit- .

                     With the Prompt Variant, you can let One Button Prompt dynamically create small variance in the workflow prompt. 0 means no effect.

                     Below here, you can generate a set of random prompts, and send them to the Workflow prompt field. The generation of the prompt uses the settings in the Main tab.
                     </font>
                     """
adv_mark1="""
                                <font size="2">
                                ### Base model will try and generate prompts fitting the selected model.
                                
                                SD1.5 --> Less natural language
                                
                                SDXL --> More natural language (default)
                                
                                Stable Cascade --> More natural language and no prompt weights
                            
                                Anime Model --> Focussed on characters and tags, adds 1girl/1boy automatically.
                            
                                ### Flufferizer

                                A simple and quick implementiation of Fooocus prompt magic
                            
                                ### Prompt enhancer

                                Choose for "superprompt-v1" to super prompt the prompts with roborovski superprompt-v1 model.
                            
                                Please install requirements for superprompt before use.

                                </font>
                                """
adv_mark2="""
                    ### Prompt compounder
                    
                    <font size="2">
                    Normally, it creates a single random prompt. With prompt compounder, it will generate multiple prompts and compound them together. 
                    
                    Keep at 1 for normal behavior.
                    Set to different values to compound that many prompts together. My suggestion is to try 2 first.
                    
                    This was originally a bug in the first release when using multiple batches, now brought back as a feature. 
                    Raised by redditor drone2222, to bring this back as a toggle, since it did create interesting results. So here it is. 
                    
                    You can toggle the separator mode. Standardly this is a comma, but you can choose an AND or a BREAK.
                    
                    You can also choose the prompt seperator mode for use with Latent Couple extension
                    
                    Example flow:

                    Set the Latent Couple extension to 2 area's (standard setting)
                    
                    In the main tab, set the subject to humanoids
                    
                    In the prefix prompt field then add for example: Art by artistname, 2 people
                    
                    Set the prompt compounder to: 2
                    
                    Set the Prompt seperator to: AND

                    Set the Prompt Seperator mode to: prefix AND prompt + suffix

                    "automatic" is entirely build around Latent Couple. It will pass artists and the amount of people/animals/objects to generate in the prompt automatically. Set the prompt compounder equal to the amount of areas defined in Laten Couple.
                    
                    Example flow:

                    Set the Latent Couple extension to 2 area's (standard setting)
                    
                    In the main tab, set the subject to humanoids
                    
                    Leave the prompt field empty
                    
                    Set the prompt compounder to: 2

                    Set the Prompt seperator to: AND

                    Set the Prompt Seperator mode to: automatic


                    </font>
                    
                    """

obp_preset_choices=[OBPresets.RANDOM_PRESET_OBP] + [OBPresets.CUSTOM_OBP] + list(OBPresets.opb_presets.keys())

def obppreset_changed(selection):
    if selection == OBPresets.CUSTOM_OBP:
      return gr.update(value="", visible=True), gr.update(visible=True), gr.update(visible=False), gr.update(value=""), gr.update(value="")
    else:
      return gr.update(visible=False), gr.update(visible=False), gr.update(visible=True), gr.update(visible=True), gr.update(visible=True)

def act_obp_preset_save(
                    obp_preset_name,
                    obp_preset_save,
                    insanitylevel,
                    subject,
                    artist,
                    chosensubjectsubtypeobject,
                    chosensubjectsubtypehumanoid,
                    chosensubjectsubtypeconcept,
                    chosengender,
                    imagetype,
                    imagemodechance,
                    givensubject,
                    smartsubject,
                    givenoutfit,
                    prefixprompt,
                    suffixprompt,
                    giventypeofimage,
                    antistring,
                ):
                    if obp_preset_name != "":
                        obp_options = OBPresets.load_obp_presets()
                        opts = {
                            "insanitylevel": insanitylevel,
                            "subject": subject,
                            "artist": artist,
                            "chosensubjectsubtypeobject": chosensubjectsubtypeobject,
                            "chosensubjectsubtypehumanoid": chosensubjectsubtypehumanoid,
                            "chosensubjectsubtypeconcept": chosensubjectsubtypeconcept,
                            "chosengender": chosengender,
                            "imagetype": imagetype,
                            "imagemodechance": imagemodechance,
                            "givensubject": givensubject,
                            "smartsubject": smartsubject,
                            "givenoutfit": givenoutfit,
                            "prefixprompt": prefixprompt,
                            "suffixprompt": suffixprompt,
                            "giventypeofimage": giventypeofimage,
                            "antistring": antistring
                        }
                        obp_options[obp_preset_name] = opts
                        OBPresets.save_obp_preset(obp_options)
                        choices = [OBPresets.RANDOM_PRESET_OBP] + list(obp_options.keys()) + [OBPresets.CUSTOM_OBP]
                        return gr.update(choices=choices, value=obp_preset_name)
                    else:
                        return gr.update()
def gen_prompt(insanitylevel, subject, artist, imagetype, antistring, prefixprompt, suffixprompt, promptcompounderlevel, seperator,givensubject,smartsubject,giventypeofimage, imagemodechance,chosengender, chosensubjectsubtypeobject, chosensubjectsubtypehumanoid, chosensubjectsubtypeconcept,givenoutfit, base_model, OBP_preset, amountoffluff, promptenhancer, presetprefix, presetsuffix,silentmode,workprompt,promptvariantinsanitylevel):

            promptlist = []

            if(silentmode==True and workprompt != ""):
                for i in range(5):
                    creative = createpromptvariant(workprompt, promptvariantinsanitylevel)
                    promptlist.append(creative)
            else:
                for i in range(5):
                    base_prompt = build_dynamic_prompt(insanitylevel,subject,artist, imagetype, False, antistring,prefixprompt,suffixprompt,promptcompounderlevel,seperator,givensubject,smartsubject, giventypeofimage, imagemodechance,chosengender, chosensubjectsubtypeobject, chosensubjectsubtypehumanoid, chosensubjectsubtypeconcept,True,False,-1,givenoutfit,False,base_model, OBP_preset, promptenhancer, "", "", presetprefix, presetsuffix)
                    fluffed_prompt = flufferizer(prompt=base_prompt, amountoffluff=amountoffluff)
                    promptlist.append(fluffed_prompt)
            return promptlist     
def prompttoworkflowprompt(text):
            return text
def OBPPreset_changed_update_custom(selection):
      # Skip if Custom was selected
      if selection == OBPresets.CUSTOM_OBP:
          return [gr.update()] * 16
  # Update Custom values based on selected One Button preset
      if selection == OBPresets.RANDOM_PRESET_OBP:
          selected_opb_preset = OBPresets.get_obp_preset("Standard")
      else:     
          selected_opb_preset = OBPresets.get_obp_preset(selection)
      return [gr.update(value=selected_opb_preset["insanitylevel"]),
      gr.update(value=selected_opb_preset["subject"]),
      gr.update(value=selected_opb_preset["artist"]),
      gr.update(value=selected_opb_preset["chosensubjectsubtypeobject"]),
      gr.update(value=selected_opb_preset["chosensubjectsubtypehumanoid"]),
      gr.update(value=selected_opb_preset["chosensubjectsubtypeconcept"]),
      gr.update(value=selected_opb_preset["chosengender"]),
      gr.update(value=selected_opb_preset["imagetype"]),
      gr.update(value=selected_opb_preset["imagemodechance"]),
      gr.update(value=selected_opb_preset["givensubject"]),
      gr.update(value=selected_opb_preset["smartsubject"]),
      gr.update(value=selected_opb_preset["givenoutfit"]),
      gr.update(value=selected_opb_preset["prefixprompt"]),
      gr.update(value=selected_opb_preset["suffixprompt"]),
      gr.update(value=selected_opb_preset["giventypeofimage"]),
      gr.update(value=selected_opb_preset["antistring"])]

def subjectsvalue(subject):
      enable=("human" in subject)
      return gr.update(visible=enable)

def subjectsvalueforsubtypeobject(subject):
      enable=(subject=="object")
      return gr.update(visible=enable)

def subjectsvalueforsubtypeconcept(subject):
      enable=(subject=="concept")
      return gr.update(visible=enable)
def subjectsvalueforsubtypeobject(subject):
      enable=(subject=="humanoid")
      return gr.update(visible=enable)

def ui():

            

                   

        



        interrupt.click(tryinterrupt, inputs=[apiurl])
        



        # turn things on and off for subject subtype object
        

        
        # turn things on and off for subject subtype humanoid
        


        # turn things on and off for subject subtype concept




        # Turn things off and on for onlyupscale and txt2img
        def onlyupscalevalues(onlyupscale):
             onlyupscale = not onlyupscale
             return {
                  amountofimages: gr.update(visible=onlyupscale),
                  size: gr.update(visible=onlyupscale),
                  samplingsteps: gr.update(visible=onlyupscale),
                  cfg: gr.update(visible=onlyupscale),

                  hiresfix: gr.update(visible=onlyupscale),
                  hiressteps: gr.update(visible=onlyupscale),
                  hiresscale: gr.update(visible=onlyupscale),
                  denoisestrength: gr.update(visible=onlyupscale),
                  upscaler: gr.update(visible=onlyupscale),

                  model: gr.update(visible=onlyupscale),
                  samplingmethod: gr.update(visible=onlyupscale),
                  upscaler: gr.update(visible=onlyupscale),

                  qualitygate: gr.update(visible=onlyupscale),
                  quality: gr.update(visible=onlyupscale),
                  runs: gr.update(visible=onlyupscale),
                  qualityhiresfix: gr.update(visible=onlyupscale),
                  qualitymode: gr.update(visible=onlyupscale),
                  qualitykeep: gr.update(visible=onlyupscale)


             }
        
        onlyupscale.change(
            onlyupscalevalues,
            [onlyupscale],
            [amountofimages,size,samplingsteps,cfg,hiresfix,hiressteps,hiresscale,denoisestrength,upscaler,model,samplingmethod,upscaler,qualitygate,quality,runs,qualityhiresfix,qualitymode,qualitykeep]
        )
        
        
        # Turn things off and on for hiresfix
        def hireschangevalues(hiresfix):
             return {
                  hiressteps: gr.update(visible=hiresfix),
                  hiresscale: gr.update(visible=hiresfix),
                  denoisestrength: gr.update(visible=hiresfix),
                  upscaler: gr.update(visible=hiresfix)
             }
        
        hiresfix.change(
            hireschangevalues,
            [hiresfix],
            [hiressteps,hiresscale,denoisestrength,upscaler]
        )

        # Turn things off and on for quality gate
        def qgatechangevalues(qualitygate):
             return {
                  quality: gr.update(visible=qualitygate),
                  runs: gr.update(visible=qualitygate),
                  qualityhiresfix: gr.update(visible=qualitygate),
                  qualitymode: gr.update(visible=qualitygate),
                  qualitykeep: gr.update(visible=qualitygate)
             }
        
        qualitygate.change(
            qgatechangevalues,
            [qualitygate],
            [quality,runs,qualityhiresfix,qualitymode,qualitykeep]
        )
        
        # Turn things off and on for USDU
        def ultimatesdupscalechangevalues(ultimatesdupscale):
             return {
                  usdutilewidth: gr.update(visible=ultimatesdupscale),
                  usdutileheight: gr.update(visible=ultimatesdupscale),
                  usdumaskblur: gr.update(visible=ultimatesdupscale),
                  usduredraw: gr.update(visible=ultimatesdupscale),

                  usduSeamsfix: gr.update(visible=ultimatesdupscale),
                  usdusdenoise: gr.update(visible=ultimatesdupscale),
                  usduswidth: gr.update(visible=ultimatesdupscale),
                  usduspadding: gr.update(visible=ultimatesdupscale),
                  usdusmaskblur: gr.update(visible=ultimatesdupscale)
             }
        
        ultimatesdupscale.change(
            ultimatesdupscalechangevalues,
            [ultimatesdupscale],
            [usdutilewidth,usdutileheight,usdumaskblur,usduredraw,usduSeamsfix,usdusdenoise,usduswidth,usduspadding,usdusmaskblur]
        )

        # Turn things off and on for EXTRAS
        def enableextraupscalechangevalues(enableextraupscale):
             return {
                  extrasupscaler1: gr.update(visible=enableextraupscale),
                  extrasupscaler2: gr.update(visible=enableextraupscale),
                  extrasupscaler2visiblity: gr.update(visible=enableextraupscale),
                  extrasresize: gr.update(visible=enableextraupscale),

                  extrasupscaler2gfpgan: gr.update(visible=enableextraupscale),
                  extrasupscaler2codeformer: gr.update(visible=enableextraupscale),
                  extrasupscaler2codeformerweight: gr.update(visible=enableextraupscale)
             }
        
        enableextraupscale.change(
            enableextraupscalechangevalues,
            [enableextraupscale],
            [extrasupscaler1,extrasupscaler2,extrasupscaler2visiblity,extrasresize, extrasupscaler2gfpgan,extrasupscaler2codeformer,extrasupscaler2codeformerweight]
        )

      

        return [insanitylevel,subject, artist, imagetype, prefixprompt,suffixprompt,negativeprompt, promptcompounderlevel, ANDtoggle, silentmode, workprompt, antistring, seperator, givensubject, smartsubject, giventypeofimage, imagemodechance, chosengender, chosensubjectsubtypeobject, chosensubjectsubtypehumanoid, chosensubjectsubtypeconcept, promptvariantinsanitylevel, givenoutfit, autonegativeprompt, autonegativepromptstrength, autonegativepromptenhance, base_model, OBP_preset, amountoffluff, promptenhancer, presetprefix, presetsuffix]
            
    

    
def run(self, p, insanitylevel, subject, artist, imagetype, prefixprompt,suffixprompt,negativeprompt, promptcompounderlevel, ANDtoggle, silentmode, workprompt, antistring,seperator, givensubject, smartsubject, giventypeofimage, imagemodechance, chosengender, chosensubjectsubtypeobject, chosensubjectsubtypehumanoid, chosensubjectsubtypeconcept, promptvariantinsanitylevel, givenoutfit, autonegativeprompt, autonegativepromptstrength, autonegativepromptenhance, base_model, OBP_preset, amountoffluff, promptenhancer, presetprefix, presetsuffix):
        
        images = []
        infotexts = []
        all_seeds = []
        all_prompts = []

        batches = p.n_iter
        initialbatchsize = p.batch_size
        batchsize = p.batch_size
        p.n_iter = 1
        p.batch_size = 1
        
        initialseed = p.seed
        if p.seed == -1:
            p.seed = int(random.randrange(4294967294))

        if(silentmode and workprompt != ""):
            print("Workflow mode turned on, not generating a prompt. Using workflow prompt.")
        elif(silentmode):
            print("Warning, workflow mode is turned on, but no workprompt has been given.")
        elif p.prompt != "" or p.negative_prompt != "":
            print("Please note that existing prompt and negative prompt fields are (no longer) used")
        
        if(ANDtoggle == "automatic" and artist == "none"):
            print("Automatic and artist mode set to none, don't work together well. Ignoring this setting!")
            artist = "all"

        #if(ANDtoggle == "automatic" and (prefixprompt != "")):
        #    print("Automatic doesnt work well if there is an prefix prompt filled in. Ignoring this prompt fields!")
        #    prefixprompt = ""

        
        state.job_count = batches
        
        for i in range(batches):
            
            if(silentmode == False):
                # prompt compounding
                print("Starting generating the prompt")
                preppedprompt = ""
                
                artistcopy = artist
                prefixpromptcopy = prefixprompt
                
                if(ANDtoggle == "automatic"):
                    if(artist != "none"):
                        preppedprompt += build_dynamic_prompt(insanitylevel,subject,artist, imagetype, True, antistring, base_model=base_model) 
                    if(subject == "humanoid"):
                        preppedprompt += ", " + promptcompounderlevel + " people"
                    if(subject == "landscape"):
                        preppedprompt += ", landscape"
                    if(subject == "animal"):
                        preppedprompt += ", " + promptcompounderlevel  + " animals"
                    if(subject == "object"):
                        preppedprompt += ", " + promptcompounderlevel  + " objects"
                    #sneaky! If we are running on automatic, we don't want "artists" to show up during the rest of the prompt, so set it to none, but only temporary!

                    artist = "none"
                

                if(ANDtoggle != "none" and ANDtoggle != "automatic"):
                    preppedprompt += prefixprompt
                
                if(ANDtoggle != "none"):
                    if(ANDtoggle!="prefix + prefix + prompt + suffix"):
                        prefixprompt = ""
                    if(seperator == "comma"):
                        preppedprompt += " \n , "
                    else:
                        preppedprompt += " \n " + seperator + " "


                #Here is where we build a "normal" prompt
                base_prompt = build_dynamic_prompt(insanitylevel,subject,artist, imagetype, False, antistring, prefixprompt, suffixprompt,promptcompounderlevel, seperator,givensubject,smartsubject,giventypeofimage,imagemodechance,chosengender, chosensubjectsubtypeobject, chosensubjectsubtypehumanoid, chosensubjectsubtypeconcept,True,False,-1,givenoutfit, False, base_model, OBP_preset, promptenhancer, "", "", presetprefix, presetsuffix)
                fluffed_prompt = flufferizer(prompt=base_prompt, amountoffluff=amountoffluff)
                preppedprompt += fluffed_prompt

                # set the artist mode back when done (for automatic mode)
                artist = artistcopy
                prefixprompt = prefixpromptcopy
                
                # set everything ready
                p.prompt = preppedprompt  
                p.negative_prompt = negativeprompt

            if(silentmode == True):
                base_prompt = createpromptvariant(workprompt,promptvariantinsanitylevel)
                p.prompt = base_prompt

            #for j in range(batchsize):
       
            print(" ")
            print("Full prompt to be processed:")
            print(" ")
            print(p.prompt)

            if(autonegativeprompt):
                 p.negative_prompt = build_dynamic_negative(positive_prompt=base_prompt, insanitylevel=autonegativepromptstrength,enhance=autonegativepromptenhance, existing_negative_prompt=p.negative_prompt, base_model=base_model)
                 

            promptlist = []
            if(batchsize>1):
            # finally figured out how to do multiple batch sizes
            
                for i in range(batchsize):
                    promptlist.append(p.prompt)

                p.prompt = promptlist
                p.batch_size = batchsize
                p.hr_prompt = promptlist
            else:
                p.batch_size = batchsize
                p.hr_prompt = p.prompt
                
            processed = process_images(p)
            images += processed.images
            infotexts += processed.infotexts
            
            # prompt and seed info for batch grid
            all_seeds.append(processed.seed)
            all_prompts.append(processed.prompt)
            
            # prompt and seed info for individual images
            all_seeds += processed.all_seeds
            all_prompts += processed.all_prompts
            
            state.job = f"{state.job_no} out of {state.job_count}" 
        
            # Increase seed by batchsize for unique seeds for every picture if -1 was chosen
            if initialseed == -1:
                p.seed += batchsize
               
        # just return all the things
        p.n_iter = 0
        p.batch_size = 0
        return Processed(p=p, images_list=images, info=infotexts[0], infotexts=infotexts, all_seeds=all_seeds, all_prompts=all_prompts)
