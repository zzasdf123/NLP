import copy

from transform_generator import *
class EncoderDecoder(nn.Module):
    def __init__(self,encoder,decoder,source_embed,target_embed,generator):
        super().__init__()
        # encoder: 编码器对象
        self.encoder = encoder
        # decoder: 解码器对象
        self.decoder = decoder
        # source_embed: 源语言的词嵌入对象
        self.source_embed = source_embed
        # target_embed: 目标语言的词嵌入对象
        self.target_embed = target_embed
        # generator: 输出层对象
        self.generator = generator

    def forward(self,source,target,source_mask1,source_mask2,target_mask):
        # source:源语言输入,形状为[batch_size,source_seq_len]——>[2,4]
        # target:目标语言输入,形状为[batch_size,target_seq_len]——>[2,6]
        # source_mask1:padding_mask,作用在编码器段多头自注意机制，形状为[head,source_seq_len,source_seq_len]——>[2,,4,4]
        # source_mask1: padding_mask, 作用在解码器段多头注意机制，形状为[head, target_seq_len, source_seq_len]——>[2,,6,4]
        # target_mask: sentence_mask, 作用在解码器段多头自注意机制，形状为[head, target_seq_len, target_seq_len]——>[2,,6,6]
        # 1.将source:源语言输入,形状为[batch_size,source_seq_len]——>[2,4]送入编码器得到——>[2,4,512]
        # word_embedding + position_embedding
        encoder_word_embed = self.source_embed(source)
        # 2.将encoder_word_embed与source_mask1送入编码器
        encoder_output = self.encoder(encoder_word_embed,source_mask1)
        # 3.将target:目标语言输入,形状为[batch_size,target_seq_len]——>[2,6]送入解码器得到——>[2,6,512]
        decoder_word_embed = self.target_embed(target)
        # 4.将encoder_output,decoder_word_embed,source_mask1,target_mask送入解码器
        decoder_output = self.decoder(decoder_word_embed,encoder_output,source_mask2,target_mask)
        # 5.将decoder_output送入输出层得到结果
        output = self.generator(decoder_output)
        return output

def ceshi_EncoderDecoder():
    # 1.实例化编码器对象
    self_attention=MultiHeadAttention(embed_dim=512,head=8,dropout_p=0.1)
    ff = FeedForward(d_model=512,d_ff=1024)
    encoder_layer = EncoderLayer(size=512,self_attention=self_attention,feed_forward=ff,dropout_p=0.1)
    encoder = Encoder(layer=encoder_layer,N=6)

    # 2.实例化解码器对象
    self_attention = copy.deepcopy(self_attention)
    src_attention = copy.deepcopy(self_attention)
    feed_forward = copy.deepcopy(ff)
    decoder_layer = DecoderLayer(size=512,self_attention=self_attention,src_attention=src_attention,feed_forward=ff,dropout_p=0.1)
    decoder = Decoder(layer=decoder_layer,N=6)

    # 3.源语言输入部分对象：word_embedding + position_embedding
    # 实例化编码器对象
    vocab_size = 1000
    d_model = 512
    encoder_embed = Embeddings(vocab_size=vocab_size, d_model=d_model)
    # 经过位置编码层
    dropout_p = 0.1
    encoder_pe = PostionalEncoding(d_model=d_model, dropout_p=dropout_p)
    source_embed = nn.Sequential(encoder_embed, encoder_pe)

    # 4.目标语言输入部分对象：word_embedding + position_embedding
    # 实例化编码器对象
    decoder_embed = copy.deepcopy(encoder_embed)
    # 经过位置编码层
    decoder_pe = copy.deepcopy(encoder_pe)
    target_embed = nn.Sequential(decoder_embed, decoder_pe)

    # 5.实例化输出对象
    generator = Generator(d_model=512,vocab_size=2000)

    # 6.实例化Encoder2Decoder对象
    transform = EncoderDecoder(encoder,decoder,source_embed,target_embed,generator)

    # 7.准备数据
    source = torch.tensor([[21,25,1,54],
                           [12,45,65,121]])
    target = torch.tensor([[21,25,1,54,12,12],
                           [12,45,65,121,32,12]])
    source_mask1 = torch.zeros(8,4,4)
    source_mask2 = torch.zeros(8,6,4)
    target_mask = torch.zeros(8,6,6)
    output = transform(source,target,source_mask1,source_mask2,target_mask)
    print(output)
    print(output.shape)

if __name__ == '__main__':
    ceshi_EncoderDecoder()