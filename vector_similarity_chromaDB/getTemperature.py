from SentenceModel import SentenceModel
from Chuncker import Chuncker
from ChromaClient import ChromaClient
from SqliteDB import SqliteDB


sql_db = SqliteDB()
chromaClient = ChromaClient()
sentenceModel = SentenceModel()
chunker = Chuncker() 


def check_news_similarity(new_text, threshold=0.8):
    """
    Обчислює косинусну схожість для кожного чанку та виносить вердикт.
    """
    # Підготовка чанків
    raw_chunks = chunker.get_chuncks(new_text)
    if not raw_chunks:
        return False

    # Перетворення на вектори (Batch Encoding)
    # Модель MiniLM видає нормалізовані вектори, що ідеально для скалярного добутку
    normalized_texts = [chunker.normalize_chunck(c) for c in raw_chunks]
    query_vectors = sentenceModel.encode_batch(normalized_texts)

    # Запит до бази
    results = chromaClient.query_collection(
        query_embeddings=query_vectors,
        n_results=1
    )

    max_sim = 0.0
    best_chunk_info = ""

    # Обчислення схожості Similarity
    if results['distances']:
        for i, dist_list in enumerate(results['distances']):
            if not dist_list: continue
            
            # Отримуємо дистанцію (d)
            distance = dist_list[0]
            
            # Обчислюємо схожість (s = 1 - d)
            similarity = 1 - distance
            
            # Логування для відладки
            print(f"--- Чанк #{i} ---")
            print(f"Відстань (L2/Cos Dist): {distance:.4f}")
            print(f"Косинусна схожість: {similarity:.4f}")

            if similarity > max_sim:
                max_sim = similarity
                best_chunk_info = raw_chunks[i]

    # Перевірка за порогом 0.8
    is_interesting = max_sim >= threshold

    print(f"\nРЕЗУЛЬТАТ:")
    print(f"Максимальна схожість у статті: {max_sim:.4f}")
    
    if is_interesting:
        print(f"✅ Новина ЦІКАВА (перевищено поріг {threshold})")
        print(f"Ключовий фрагмент: {best_chunk_info[:100]}...")
    else:
        print(f"❌ Новина НЕ цікава (нижче порогу {threshold})")

    return is_interesting


data_test = [
    """Polițiștii de frontieră din cadrul Inspectoratului Teritorial al Poliției de Frontieră Giurgiu – Serviciul Teritorial al Poliției de Frontieră Dolj au depistat în urma unor acțiuni de tip BLITZ, 250.000 de țigarete, pe care un cetățean român a încercat să le introducă în țară fără documente legale. A fost nevoie de unelte specifice pentru deschiderea și demontarea controlată a compartimentului marfă în care se aflu ascunse țigările.
    În data de 02.03.2026, în jurul orei 05.15,  polițiștii de frontieră au oprit pentru control un microbuz frigorific înmatriculat în România, condus de un cetățean român în vârstă de 28 de ani, care transporta, conform documentelor, o hotă industrială.
    La o verificare preliminară a mijlocului de transport, polițiștii de frontieră au observat că ușa laterală a compartimentului de marfă nu permitea accesul în interior. În urma unor verificări suplimentare, polițiștii au descoperit că în interior se află o  cantitate semnificativă de țigări, în urma inventarierii rezultând în total de 250.000 de țigarete (12.500 de pachete), fără timbru fiscal, pentru care persoana în cauză nu deținea documente de proveniență.
    Întreaga cantitate de țigarete, estimată la o valoare de peste 250.000 de lei, precum și mijlocul de transport, evaluat la peste 60.000 de lei, au fost ridicate.
    Între timp, meteorologii anunță că vremea în sudul României se va menține neobișnuit de caldă pentru începutul lunii martie. 
    Temperaturile vor atinge 18°C, iar cerul va fi variabil, cu șanse reduse de precipitații în zona punctelor de frontieră Giurgiu și Calafat. 
    Vântul va sufla slab până la moderat, favorizând condițiile de drum pe timp de zi.
    Revenind la cazul de contrabandă, întreaga cantitate de țigarete, estimată la 250.000 de lei, precum și mijlocul de transport, au fost ridicate în vederea confiscării.
    """,
    """Renault România a fost amendată cu 125.000 de euro după ce, în urma unui atac cibernetic asupra unei aplicații a companiei care era gestionată printr-un împuternicit, datele personale ale unui număr foarte mare de persoane au fost accesate și divulgate în mod neautorizat prin publicarea pe o platformă, a anunțat miercuri Autoritatea Națională pentru Protecția Datelor Personale (ANSPDCP).\n Investigația a fost demarată după ce Renault Commercial Roumanie SRL (n.a. Renault România) a notificat autoritatea cu privire la încălcarea securității datelor cu caracter personal, potrivit dispozițiilor art. 33 din Regulamentul (UE) 2016/679.""",
    """Nicușor Dan și Volodimir Zelenski au semnat joi, la Palatul Cotroceni, documentele necesare astfel încât România și Ucraina să devină parteneri strategici, cel mai înalt nivel de colaborare între două state.
    Într-o conferință de presă, la finalul evenimentului, președintele Nicușor Dan a subliniat că prin această declarație statele își „asumă încrederea reciprocă, responsabilitatea comună pentru această parte de Europa, pentru cetățenii ei și întreaga regiune”.
    El a făcut referire și la tensiunile bilaterale din trecut. „Nu trebuie să ne ascundăm să spunem că istoric a existat neîncredere între țările noastre. Această neîncredere s-a evaporat în momentul începerii războiului din 2022”.
    Unele din aceste tensiuni vizau drepturile etnicilor români de pe teritoriul Ucrainei, subiect abordat de președintele român. „Ucraina are o reformă a educației. Prin perspectiva acestui document face o excepție pentru minorități ca să-și poată desfășura educația în localitatea lor în limba maternă. Aceste garanții oferite minorităților funcționează”.
    La rândul său, președintele Volodimir Zelenski a mulțumit României pentru sprijinul acordat în cei patru ani de război. „România și alte state ne-au ajutat să ne apărăm cetățenii. Mulțumim pentru componenta de protejare împotriva rachetelor”, a replicat liderul de la Kiev.
    În ceea ce privește dezvoltarea de drone, unde Ucraina are o expertiză importantă acumulată în cei patru ani de război, Zelenski s-a arătat deschis pentru o colaborare. „Suntem vecini buni și putem dezvolta parteneriate de fabricație la fel cum am făcut în Germania, Danemarca. Aceste capacități sunt foarte utile. Cu siguranță, suntem deschiși pentru România”.""",
    """Forţele Aeriene Române au început luni o nouă misiune de Poliţie Aeriană Întărită în spaţiul aerian baltic, în cadrul angajamentului României faţă de securitatea colectivă a NATO. Detaşamentul „Carpathian Vipers", alcătuit din aproximativ 100 de militari cu şase aeronave de luptă F-16 Fighting Falcon, este dislocat în Lituania, în baza aeriană Šiauliai, şi va asigura serviciul de poliţie aeriană în perioada aprilie – iulie 2026, anunță MApN. Misiunea militarilor români este protejarea integrităţii spaţiului aerian al țărilor baltice şi reprezintă o componentă fundamentală a angajamentului NATO faţă de membrii săi. „Avioanele de luptă F-16 Fighting Falcon ale Forţelor Aeriene Române asigură permanent avertizarea timpurie şi intervenţia pentru clarificarea situaţiei aeriene, aplicând măsurile legale împotriva aeronavelor care utilizează neautorizat spaţiul aerian al Ţărilor Baltice”, precizează comunicatul citat de News.ro. Forţele Aeriene Române se află la a patra dislocare în Lituania, unde au asigurat protejarea integrităţii spaţiului aerian al Ţărilor Baltice şi al NATO. Prima misiune de Poliţie Aeriană a fost executată în perioada august-octombrie 2007, cu un detaşament format din 67 de militari şi patru MiG -21 LanceR din Baza 71 Aeriană „General Emanoil Ionescu". Cea de-a doua şi cea de-a treia misiunea au fost executate în perioada aprilie-iulie 2023, respectiv aprilie-iulie 2025, cu câte 100 de militari şi patru F-16 Fighting Falcon din Baza 86 Aeriană „Locotenent Aviator Gheorghe Mociorniţă".""",
    """Conform sondajului Avangarde, la întrebarea „Din punctul dumneavoastră de vedere Armata Română are capacitatea de a rezista 48 de ore, în cazul unui atac, aşa cum se specifică în regulamentul NATO?”, 49% au răspuns negativ, 35% au răspuns afirmativ şi 16% nu ştiu sau nu răspund. De asemenea, 58% se declară de acord cu solicitarea NATO de creştere a cheltuielilor pentru înarmare a tuturor ţărilor membre, deci şi a României, iar 30% nu sunt de acord cu acest lucru. Totodată, 48% consideră că nu ar fi o idee bună reintroducerea stagiului militar obligatoriu, în timp ce 43% afirmă că ar fi o idee bună. La întrebarea „Dumneavoastră personal consideraţi că în cazul în care drone fără pilot mai intră în spaţiul aerian al României ar trebui ca acestea să fie doborâte?”, 44% au răspus „cu siguranţă da”, 41% - „în funcţie de situaţie”, 8% au răspuns „mai degrabă nu” şi 2% „cu siguranţă nu”. Conform sondajului, 42% afirmă că, în prezent, pericolul ca Rusia să atace intenţionat România este mic, iar 16% cred că acest risc este foarte mic sau nu există. În schimb, 4% cred că pericolul este foarte mare, iar 29% consideră că acesta este mare. De asemenea, 58% consideră că Guvernul României gestionează prost capitolul Apărării Naţionale şi 22% cred că Executivul face bine acest lucru. La întrebarea „În situaţia în care Federaţia Rusă ar ataca Republica Moldova credeţi că România ar trebui să trimită trupe pentru a o apăra?”, 55% au răspuns negativ, 28% au dat un răspuns afirmativ, iar 17% au afirmat că nu ştiu sau nu au răspuns. Potrivit aceluiaşi sondaj, 75% consideră că, având în vedere situaţia geo-militară a României din această perioadă, Ministerul Apărării Naţionale ar trebui să fie condus de „un ofiţer superior în retragere cu calităţi profesionale recunoscute”, iar 11% - de un civil. Sondajul Avangarde „Percepţii asupra capacităţilor Armatei Române” a fost realizat în perioada 6-10 octombrie, telefonic, pe un eşantion de 920 de persoane. Eroarea maximă de eşantionare, la un nivel de încredere de 95%, este de +/- 3,4%.""",
    """Ministerul Apărării anunță miercuri că, în noaptea de 9 spre 10 septembrie, un grup de drone a fost detectat în zona localității Vâlcov din Ucraina, la granița cu România. Armata a ridicat două aeronave de luptă F-16 pentru misiuni de cercetare iar în timpul nopții a fost transmis un mesaj Ro-Alert pentru zona de nord a județului Tulcea. Nu au fost detectate pătrunderi în spațiul aerian al României.  În noaptea de 9 spre 10 septembrie, sistemele de supraveghere radar ale MApN au detectat un grup de drone aeriene, în zona localității ucrainene Vâlcov, la granița cu România, arată un comunicat de presă al MApN.  Două aeronave de luptă F-16 ale Forțelor Aeriene Române au decolat la ora 00:59 de la Baza 86 Aeriană din Feteşti pentru misiuni de cercetare.   La ora 01:27 a fost transmis mesajul Ro-Alert pentru zona de nord a județului Tulcea.   Nu au fost detectate pătrunderi ale unor vehicule aeriene în spațiul aerian național.   La ora 02:35 a fost transmis semnalul de încetare a alertei aeriene, iar aeronavele au revenit în baza de dislocare.   „Ministerul Apărării Naționale menține în permanență un nivel ridicat de vigilență și asigură supravegherea strictă a spațiului aerian, maritim și terestru national. Suntem în contact permanent cu aliații noștri, schimbăm în timp real informații operative și acționăm ferm pentru garantarea securității României și a flancului estic al NATO”, arată sursa citată.""",
    """Compania germană producătoare armament Rheinmetall a anunțat, luni, că va produce transportoare blindate Lynx în România, la Automecanica Mediaș. Compania își va extinde astfel prezența în țară și rolul de producător și furnizor pentru Armata română. Un nou Centru de Excelență Rheinmetall în România se va concentra pe transferul de know-how tehnic către forța de muncă locală. Echipat cu simulatoare avansate și programe practice de instruire, centrul va oferi expertiză în operarea și asistența sistemelor precum vehiculul Lynx. """,
    """„Descalificați” din programul armei de asalt? Beretta spune că România nu i-a cerut oferta pentru puștile de asalt NARP și pistoalele APX iar Bucureștiul îi „respinge apelul”. DefenseRomania a stat de vorbă în exclusivitate cu Carlo Ferlito, CEO Beretta, despre achiziția viitoarelor arme de asalt pentru Armata României și oferta grupului italian, prilej cu care CEO-ul companiei nu și-a ascuns „dezamăgirea” cu privire la faptul că Beretta nu a mai fost contactată de România în vederea programului, de aproape patru luni.""",
    """Bogdan Ochiană, CEO Orbotix și Ion Mocanu, Co-founder & CEO Qognifly au oferit un interviu DefenseRomania în care au anunțat un consorțiu strategic pentru apărarea autonomă. Cele două companii românești și-au propus să construiască împreună în România soluții accesibile în ceea ce privește tehnologiile emergente, drone și apărarea împotriva dronelor. Qognifly a dezvoltat deja soluția Wall Drone de combatere a dronelor, o soluție românească care are componente combat proven în Ucraina. Totodată, compania are memorandumuri cu companii ucrainene și livrează deja în 16 țări. În baza acestor experiențe, noul consorțiu își propune să ducă cooperarea la următorul nivel.""",
    """Prima știre are legătură cu achizitia a 24 de mașini blindate 4×4 VAMTAC ST5 BN2, contra unei sume de 416 mii de euro per bucată. Mașinile vor fi alocate Bazei 3 Logistice “Zargidava” din Roman.
    Urovesa VAMTEC ST5 4×4 este o mașină blindată 4×4 bună la toate și poate fi configurată pentru a îndeplini o multitudine de roluri. Armata Română este familiară cu mașinile VAMTAC, existând deja în dotare astfel de mașini în număr de aproximativ 70 de unități.
    Din păcate, sau din prostia noastră clasică, nu s-a mers pe uniformizarea flotei de blindate 4×4 și licitația pentru 1059 de astfel de mașini a fost câștigate de un alt furnizor. Nu spun că Otokar se ridică sau nu la calitatea și seriozitatea celor de la Urovesa, dar spun că în materie de mașini blindate sau mai puțin blindate 4×4 avem o adevărată menajerie, sau grădina zoologică: multe modele în numere mici.
    Avem Humvee – în jur de 300, Uro Vamtac în jur de 70+alte 24 contractate acum, avem alea 16 Panhard de la Comandamentul Logistic, L-ATV – 129 contractate pentru FOS, Technamm (Toyota Land Cruiser modificată) pentru FOS. Și ajungem la știrea a adoua.""",
]

for news in data_test:
    print("\n\n===============================")
    check_news_similarity(news)

