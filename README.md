# berry_measuring

Per foto
- detectie van kruispunten ruitjespapier -> grid_spacing (mm/pixel)
- segmentatie van bes -> masker
- ellips fitten op masker -> grote en kleine as (mm)

Per bes
- r1 = max van alle grote assen
- r2 = max van alle kleine assen
- r3 = min van alle kleine assen
- vol = 4*pi/3*r1-r2*r3

Outputs
- Plot per foto: foto zelf, segmentatie, ellipsfit met assen (bes.../....jpg)
- Tabel met resultaten per bes (results.xlsx)
