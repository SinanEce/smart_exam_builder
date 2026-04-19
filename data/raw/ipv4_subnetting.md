# IPv4 Alt Ağlama

IPv4 adresleri 32 bitten oluşur ve genellikle noktalı onluk gösterimle yazılır. Bir ağın hangi bölümünün ağ adresini, hangi bölümünün host adresini gösterdiği alt ağ maskesi ile belirlenir. Örneğin 192.168.10.0/24 ağında ilk 24 bit ağ kısmıdır ve kalan 8 bit host kısmıdır.

Alt ağlama, büyük bir IP bloğunu daha küçük mantıksal ağlara bölme işlemidir. Bu işlem yayın trafiğini sınırlandırır, adres kullanımını düzenler ve ağ yönetimini kolaylaştırır. CIDR gösteriminde /26 maskesi 255.255.255.192 anlamına gelir. Bu maske ile son oktette 64 adreslik bloklar oluşur: 0, 64, 128 ve 192.

Bir alt ağda kullanılabilir host sayısı, host bit sayısına göre 2^n - 2 formülüyle hesaplanır. Çıkarılan iki adres ağ adresi ve yayın adresidir. Örneğin /26 maskesinde 6 host biti vardır, bu nedenle her alt ağda 62 kullanılabilir host adresi bulunur.

Alt ağ sorularında önce maske uzunluğu belirlenir, sonra blok boyutu hesaplanır. IP adresinin hangi blok aralığına düştüğü bulunur. Son olarak ağ adresi, ilk kullanılabilir adres, son kullanılabilir adres ve yayın adresi belirlenir.

