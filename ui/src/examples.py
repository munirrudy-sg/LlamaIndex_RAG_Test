query = [
    {"input": "daftar customer/nasabah vip", 
     "query": "SELECT * FROM cc_trx WHERE `Revw Desc`='VIP';"
    },
    {"input": "daftar customer/nasabah vvip", 
     "query": "SELECT * FROM cc_trx WHERE `Revw Desc`='VVIP';"
    },
    {
        "input": "daftar customer/nasabah yang merupakan karyawan bank sinarmas",
        "query": "SELECT * FROM cc_trx WHERE `Revw Desc`='Krywn Bank Sinarmas';"
    },
    {
        "input": "daftar customer/nasabah yang merupakan karyawan group/grup sinarmas",
        "query": "SELECT * FROM cc_trx WHERE `Revw Desc`='Krywn Group Sinarmas';"
    },
    {
        "input": "daftar customer/nasabah yang merupakan korporasi atau perusahaan",
        "query": "SELECT * FROM cc_trx WHERE `Revw Desc`='CORPORATE';"
    },
    {
        "input": "Mengecek tipe kartu corporate atau transaksi menggunakan kartu corporate",
        "query": "SELECT * FROM cc_trx WHERE `Card Type (group)`='Corporate';"
    },
    {
        "input": "Mengecek tipe kartu grup individu",
        "query": "SELECT * FROM cc_trx WHERE `Card Type (group)`='Individu';"
    },
    {
        "input": "customer/nasabah yang telah melakukan transaksi",
        "query": "SELECT * FROM cc_trx where `Jumlah Trx` != '';"
    },
    {
        "input": "customer/nasabah yang tidak melakukan transaksi sama sekali",
        "query": "SELECT * FROM cc_trx where `Jumlah Trx` = '';"
    },
    {
        "input": "menghitung jumlah transaksi",
        "query": "SELECT SUM(`Jumlah Trx`) FROM cc_trx WHERE `Jumlah Trx` != ''"
    },
    {
        "input": "berapa jumlah transaksi berdasarkan kategori nasabah",
        "query": "SELECT `Revw Desc`,SUM(`Jumlah Trx`) FROM cc_trx GROUP BY `Revw Desc`;"
    },
    {
        "input": "berapa jumlah transaksi secara nominal",
        "query": "SELECT SUM(`Nominal`) FROM cc_trx"
    },
    {
        "input": "berapa jumlah transaksi secara nominal berdasarkan kategori nasabah/customer",
        "query": "SELECT `Revw Desc`,SUM(`Nominal`) FROM cc_trx GROUP BY `Revw Desc`;"
    },
    {
        "input": "daftar area / wilayah sales and distribution (SND)",
        "query": "SELECT DISTINCT `Area` FROM cc_trx"
    },

    {
        "input": "kategori nasabah yang memiliki jumlah transaksi terbanyak di bulan juni",
        "query": "SELECT `Revw Desc` , SUM(`Jumlah Trx`) FROM db_cc.cc_trx WHERE MONTH(`Report Date`) = 6 GROUP BY `Revw Desc` order by SUM(`Jumlah Trx`) DESC;"
    },
    
    {
        "input": "data transaksi customer/nasabah umum",
        "query": "SELECT * FROM cc_trx WHERE `Revw Desc`='Umum';"
    },
    {
        "input": "menghitung jumlah customer/nasabah berdasarkan tipe card holder",
        "query": "SELECT `Revw Desc` as `user group`, COUNT(*)  FROM cc_trx GROUP BY `Revw Desc`"
    },
    {
        "input": "cek transaksi di tahun 2024",
        "query": "SELECT * FROM cc_trx WHERE YEAR(`Report Date`) = '2024'"
    },
    {
        "input": "cek transaksi di bulan juli",
        "query": "SELECT * FROM cc_trx WHERE MONTH(`Report Date`) = '7'"
    },
    {
        "input": "daftar kantor cabang / anak cabang bank sinarmas",
        "query": "SELECT DISTINCT Branch FROM cc_trx"
    },
    {
        "input": "daftar kantor parent/induk cabang bank sinarmas",
        "query": "SELECT DISTINCT `Parent Branch` FROM cc_trx"
    },
    {
        "input": "daftar wilayah atau region kantor cabang",
        "query": "SELECT DISTINCT Region FROM cc_trx"
    },
    {
        "input": "daftar tier limit kartu kredit",
        "query": "SELECT DISTINCT `Tier Limit` FROM cc_trx"
    },
    {
        "input": "kebanyakan customer/nasabah menggunakan kartu kredit untuk belanja apa",
        "query": "SELECT `Mcc Desc`, COUNT(*) FROM cc_trx WHERE `Mcc Desc` != ''  GROUP BY `Mcc Desc` ORDER BY COUNT(*) DESC"
    },
    {
        "input": "customer/nasabah yang transaksi menggunakan kartu kredit utama",
        "query": "SELECT * FROM cc_trx WHERE `Card Type Ps` = 'Primary'"
    },
    {
        "input": "customer/nasabah yang transaksi menggunakan kartu kredit tambahan",
        "query": "SELECT * FROM cc_trx WHERE `Card Type Ps` = 'Supplement'"
    },
    {
        "input": "customer/nasabah yang transaksi menggunakan secured kartu kredit",
        "query": "SELECT * FROM cc_trx WHERE `Sec Cc Acct Typ` = 'Secured'"
    },
    {
        "input": "customer/nasabah yang transaksi menggunakan tidak secured kartu kredit",
        "query": "SELECT * FROM cc_trx WHERE `Sec Cc Acct Typ` = 'Unsecured'"
    },
    {
        "input": "kelompok umur nasabah ada apa aja?",
        "query": "SELECT DISTINCT `Tier Age` FROM cc_trx"
    },
    {
        "input": "nasabah yang berumur kurang dari 21 tahun",
        "query": "SELECT * FROM cc_trx WHERE `Tier Age` = '1. < 21'"
    },
    {
        "input": "nasabah yang berumur 31 sampai 40 tahun",
        "query": "SELECT * FROM cc_trx WHERE `Tier Age` = '3. 31 s.d 40'"
    },
    {
        "input": "nasabah yang berumur 41 sampai 50 tahun",
        "query": "SELECT * FROM cc_trx WHERE `Tier Age` = '4. 41 s.d 50'"
    },
    {
        "input": "nasabah yang berumur 21 sampai 30 tahun",
        "query": "SELECT * FROM cc_trx WHERE `Tier Age` = '2. 21 s.d 30'"
    },
    {
        "input": "nasabah yang berumur 51 sampai 60 tahun",
        "query": "SELECT * FROM cc_trx WHERE `Tier Age` = '5. 51 s.d 60'"
    },
    {
        "input": "nasabah yang berumur lebih dari 60 tahun",
        "query": "SELECT * FROM cc_trx WHERE `Tier Age` = '6. > 60'"
    },
    {
        "input": "jumlah transaksi nasabah",
        "query": "SELECT SUM(`Jumlah Trx`) FROM cc_trx"
    },
    {
        "input": "daftar nasabah yang melakukan transaksi international atau luar negeri",
        "query": "SELECT * FROM db_cc.cc_trx WHERE `Flag Location Trx`='International'"
    },
    {
        "input": "jumlah transaksi international atau luar negeri",
        "query": "SELECT SUM(`Jumlah Trx`) FROM db_cc.cc_trx WHERE `Flag Location Trx`='International'"
    },
    {
        "input": "jumlah transaksi domestik atau dalam negeri yang menggunakan EDC bank sinarmas",
        "query": "SELECT SUM(`Jumlah Trx`) FROM db_cc.cc_trx WHERE `Flag Location Trx`='Domestic On Us'"
    },
    {
        "input": "jumlah transaksi domestik atau dalam negeri yang menggunakan EDC bank lain atau bukan EDC bank sinarmas",
        "query": "SELECT SUM(`Jumlah Trx`) FROM db_cc.cc_trx WHERE `Flag Location Trx`='Domestic Off Us'"
    },
    {
        "input": "jumlah transaksi domestik atau dalam negeri",
        "query": "SELECT SUM(`Jumlah Trx`) FROM db_cc.cc_trx WHERE `Flag Location Trx`!='International'"
    },
    {
        "input": "nasabah atau customer yang melakukan transaksi online",
        "query": "SELECT * FROM db_cc.cc_trx WHERE `Flag Digital`='Digital'"
    },
    {
        "input": "nasabah atau customer yang melakukan transaksi offline",
        "query": "SELECT * FROM db_cc.cc_trx WHERE `Flag Digital`='Non Digital'"
    },
    {
        "input": "jumlah transaksi online",
        "query": "SELECT SUM(`Jumlah Trx`) FROM db_cc.cc_trx WHERE `Flag Digital`='Digital'"
    },
    {
        "input": "jumlah transaksi offline",
        "query": "SELECT SUM(`Jumlah Trx`) FROM db_cc.cc_trx WHERE `Flag Digital`='Non Digital'"
    },
    {
        "input": "data customer atau nasabah yang menggunakan tipe kartu korporat",
        "query": "SELECT * FROM db_cc.cc_trx WHERE `Card Type`='Corporate'"
    },
    {
        "input": "data customer atau nasabah yang menggunakan tipe kartu indigo",
        "query": "SELECT * FROM db_cc.cc_trx WHERE `Card Type`='Indigo'"
    },
    {
        "input": "data customer atau nasabah yang menggunakan tipe kartu platinum",
        "query": "SELECT * FROM db_cc.cc_trx WHERE `Card Type`='Platinum'"
    },
    {
        "input": "data customer atau nasabah yang menggunakan tipe kartu alfamart",
        "query": "SELECT * FROM db_cc.cc_trx WHERE `Card Type`='Alfamart'"
    },
    {
        "input": "data customer atau nasabah yang menggunakan tipe kartu orami",
        "query": "SELECT * FROM db_cc.cc_trx WHERE `Card Type`='Orami'"
    },
    {
        "input": "melihat kode unik customer atau nomor customer",
        "query": "SELECT DISTINCT CIF_NUMBER FROM db_cc.cc_trx"
    },
    {
        "input": "customer atau nasabah yang memiliki tier limit lebih dari 100 juta",
        "query": "SELECT * FROM db_cc.cc_trx WHERE `Tier Limit` = '8. > 100 Jt'"
    },
    {
        "input": "customer atau nasabah yang memiliki tier limit 5 juta atau kurang dari 5 juta",
        "query": "SELECT * FROM db_cc.cc_trx WHERE `Tier Limit` = '1. 0 s.d 5 Jt'"
    },
    {
        "input": "customer atau nasabah yang memiliki tier limit lebih dari 40 juta sampai dengan 50 juta",
        "query": "SELECT * FROM db_cc.cc_trx WHERE `Tier Limit` = '6. > 40 Jt s.d 50 Jt'"
    },
    {
        "input": "customer atau nasabah yang memiliki tier limit lebih dari 30 juta sampai dengan 40 juta",
        "query": "SELECT * FROM db_cc.cc_trx WHERE `Tier Limit` = '5. > 30 Jt s.d 40 Jt'"
    },
    {
        "input": "customer atau nasabah yang memiliki tier limit lebih dari 10 juta sampai dengan 20 juta",
        "query": "SELECT * FROM db_cc.cc_trx WHERE `Tier Limit` = '3. > 10 Jt s.d 20 Jt'"
    },
    {
        "input": "customer atau nasabah yang memiliki tier limit lebih dari 20 juta sampai dengan 30 juta",
        "query": "SELECT * FROM db_cc.cc_trx WHERE `Tier Limit` = '4. > 20 Jt s.d 30 Jt'"
    },
    {
        "input": "customer atau nasabah yang memiliki tier limit lebih dari 5 juta sampai dengan 10 juta",
        "query": "SELECT * FROM db_cc.cc_trx WHERE `Tier Limit` = '2. > 5 Jt s.d 10 Jt'"
    },
    {
        "input": "customer atau nasabah yang memiliki tier limit lebih dari 50 juta sampai dengan 100 juta",
        "query": "SELECT * FROM db_cc.cc_trx WHERE `Tier Limit` = '7. > 50 Jt s.d 100 Jt'"
    },
    {
        "input": "berapa jumlah transaksi tier limit lebih dari 100 juta selama tahun 2024",
        "query": "SELECT COUNT(*) FROM cc_trx WHERE `Tier Limit` = '8. > 100 Jt' AND YEAR(`Report Date`) = '2024'"
    }
]