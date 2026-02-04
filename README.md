# Ambitus Analyzer - Patkányviselkedés Elemző Rendszer

Automatizált videóelemző rendszer patkányok térbeli viselkedésének nyomonkövetésére és elemzésére kísérleti környezetben.

## 📋 Áttekintés

Az Ambitus Analyzer egy Python-alapú számítógépes látás alkalmazás, amely videófelvételekről automatikusan azonosítja és követi a patkányok mozgását, rögzíti a különböző területeken töltött időt, valamint elemzi a viselkedési mintázatokat. A rendszer háttérkivonás alapú objektumdetektálást és konvexhéj alapú pozíciókövetést alkalmaz.

## ✨ Főbb Funkciók

- **Automatikus objektumkövetés**: Valós idejű patkánydetektálás háttérkivonással (MOG2)
- **Interaktív területkijelölés**: Grafikus felület folyosók, ablakok és jutalmak megjelöléséhez
- **Többféle régió támogatása**:
  - 4 folyosó (poligon alakú területek)
  - 8 ablak (téglalap alakú területek)
  - Egyéni számú jutalom pozíció
- **Robusztus pozíciókövetés**: Becsült pozíció megőrzése átmeneti elvesztés esetén
- **Középpont-detektálás**: Folyosó középpontok automatikus áthaladás-érzékelése
- **Debug mód**: Vizuális visszajelzés a követési folyamatról
- **CSV exportálás**: Részletes adatok képkockánkénti bontásban

## 🛠️ Telepítés

### Előfeltételek

- Python 3.7+
- OpenCV (cv2)
- NumPy

### Telepítési Lépések

```bash
pip install opencv-python numpy
```

## 🚀 Használat

### Alapvető Futtatás

```bash
python ambitus_analyzer.py
```

### Lépések

1. **Debug mód választása**: Döntsd el, szeretnéd-e látni a követési folyamatot valós időben
   - `y` = vizuális megjelenítés
   - `n` = háttérben futás (gyorsabb)

2. **Jutalmak számának megadása**: Add meg, hány jutalom pozíciót szeretnél megjelölni

3. **Területek kijelölése**:
   - **Ablakok**: Kattints és húzd az egeret téglalap rajzolásához
   - **Folyosók**: Kattints 4 pontot a poligon sarkaira
   - **Jutalmak**: Téglalap alakú területek megjelölése

4. **Billentyűparancsok**:
   - `n` = következő mód (ablak → folyosó → jutalom)
   - `z` = utolsó terület törlése
   - `c` = elemzés indítása
   - `q` = kilépés

### Kimenet

Az elemzés 3 CSV fájlt hoz létre:

1. **`{video_név}.csv`**: Képkockánkénti jelenlét minden területen (bináris)
2. **`{video_név}_continuous_positions.csv`**: Folyamatos pozíció adatok (x, y koordináták, becslés jelző)
3. **`{video_név}_speed_recording.csv`**: Folyosó középpont áthaladási események

## 📊 Vizualizációs Példák

A `Vizualizáció/` mappa tartalmazza az elemzési eredményeket:

- **Pozíció hőtérképek**: Az állatok mozgási mintázatai
- **Jelenlét tortadiagramok**: Folyosókban és ablakokban töltött idő aránya
- **Aktivitás összehasonlítás**: Különböző kísérletek összehasonlítása
- **Jutalom események**: Jutalmak begyűjtésének időpontjai
- **Középpont áthaladások**: Folyosóban való áthaladások száma

## 📁 Fájlstruktúra

```
szakdolgozat/
├── ambitus_analyzer.py          # Fő elemző szkript
├── regions.csv                  # Mentett területek (generált)
├── LE_17_1_64.mpg              # Bemeneti videók
├── LE_17_1_64.csv              # Kimeneti adatok
├── LE_17_1_64_continuous_positions.csv
├── LE_17_1_64_speed_recording.csv
├── Dokumentumok/                # Dokumentációk
└── Vizualizáció/               # Generált grafikonok
```

## ⚙️ Konfigurációs Paraméterek

Az `analyze_video()` függvényben módosíthatók:

- `VAR_THRESHOLD`: Háttérkivonás érzékenysége (alapértelmezett: 60)
- `MIN_AREA`: Minimális konvex terület a detektáláshoz (alapértelmezett: 450 pixel)
- `MAX_LOST_FRAMES`: Max képkocka becslési módban (alapértelmezett: 1000)
- `WARMUP_FRAMES`: Háttérmodell bemelegítési képkockák (alapértelmezett: 100)
- `MIDPOINT_THRESHOLD`: Középpont közelség távolsága (alapértelmezett: 30 pixel)

## 🧪 Algoritmus Működése

### 1. Háttérkivonás
A MOG2 (Mixture of Gaussians v2) algoritmus folyamatosan frissülő háttérmodellt épít, amely lehetővé teszi a mozgó objektumok detektálását (a patkány) a statikus hátteren.

### 2. Pozíciókövetés
- **Aktív követés**: Kontúr centroidja alapján
- **Becsült követés**: Ha a detektálás átmenetileg megszakad, az utolsó ismert pozíció kerül rögzítésre

### 3. Térdetektálás
- **Folyosók**: Poligonon belüli pont-teszt (OpenCV `pointPolygonTest`)
- **Ablakok/Jutalmak**: Téglalapon belüli pont-teszt

### 4. Középpont-áthaladás
Minden folyosóhoz tartozik egy központi pont. Ha a patkány `MIDPOINT_THRESHOLD` távolságon belülre kerül, áthaladási esemény rögzítésre kerül.

## 📌 Példa Használati Eset

Egy patkány labirintus kísérlet elemzése, ahol:
- 4 folyosó összeköttetést biztosít
- 8 ablak pozíciót jelöl
- 3 jutalom pontot helyezünk el

A rendszer automatikusan rögzíti:
- Mikor tartózkodik az állat melyik területen
- Mennyi időt tölt az egyes területeken
- Mikor gyűjti be a jutalmakat
- Milyen útvonalakat követ (középpont-áthaladások)

## 🤝 Hozzájárulás

Ez egy szakdolgozati projekt. Kérdések vagy javaslatok esetén nyiss issue-t vagy pull requestet.

## 📝 Licenc

Ez a projekt oktatási célokat szolgál.

## 👨‍💻 Szerző

Készült egyetemi szakdolgozat keretében - 2026

---

**Megjegyzés**: A program `.mpg` formátumú videófájlokat dolgoz fel. Győződj meg róla, hogy a videófájlok az `ambitus_analyzer.py` szkripttel azonos mappában találhatók, vagy frissítsd a `video_path` változót a szkriptben.
