# Copyright 2022 Proyectos y Sistemas de Mantenimiento SL (eProsima).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
ImageData Publisher
"""
from threading import Condition
import time

import fastdds
import MultiCameraData
import numpy as np
import cv2

DESCRIPTION = """ImageData Publisher example for Fast DDS python bindings"""
USAGE = ('python3 MultiCameraDataPublisher.py')

class WriterListener (fastdds.DataWriterListener) :
    def __init__(self, writer) :
        self._writer = writer
        super().__init__()


    def on_publication_matched(self, datawriter, info) :
        if (0 < info.current_count_change) :
            print ("Publisher matched subscriber {}".format(info.last_subscription_handle))
            self._writer._cvDiscovery.acquire()
            self._writer._matched_reader += 1
            self._writer._cvDiscovery.notify()
            self._writer._cvDiscovery.release()
        else :
            print ("Publisher unmatched subscriber {}".format(info.last_subscription_handle))
            self._writer._cvDiscovery.acquire()
            self._writer._matched_reader -= 1
            self._writer._cvDiscovery.notify()
            self._writer._cvDiscovery.release()


class Writer:


    def __init__(self):
        self._matched_reader = 0
        self._cvDiscovery = Condition()
        self.index = 0

        factory = fastdds.DomainParticipantFactory.get_instance()
        self.participant_qos = fastdds.DomainParticipantQos()
        transit_qos = fastdds.TransportConfigQos()
        transit_qos.send_socket_buffer_size = 12582912
        self.participant_qos.transport(transit_qos)


        factory.get_default_participant_qos(self.participant_qos)
        self.participant = factory.create_participant(0, self.participant_qos)

        self.topic_data_type = MultiCameraData.ImageDataPubSubType()
        self.topic_data_type.set_name("ImageData")
        self.type_support = fastdds.TypeSupport(self.topic_data_type)
        self.participant.register_type(self.type_support)

        self.topic_qos = fastdds.TopicQos()
        self.participant.get_default_topic_qos(self.topic_qos)
        self.topic = self.participant.create_topic("ImageDataTopic", self.topic_data_type.get_name(), self.topic_qos)

        self.publisher_qos = fastdds.PublisherQos()
        self.participant.get_default_publisher_qos(self.publisher_qos)
        self.publisher = self.participant.create_publisher(self.publisher_qos)

        self.listener = WriterListener(self)
        self.writer_qos = fastdds.DataWriterQos()
        self.publisher.get_default_datawriter_qos(self.writer_qos)

        # self.writer_qos.history = fastdds.KEEP_ALL_HISTORY_QOS
        self.writer_qos.history().kind = fastdds.KEEP_ALL_HISTORY_QOS
        # self.writer_qos.reliability = fastdds.RELIABLE_RELIABILITY_QOS
        self.writer_qos.reliability().kind = fastdds.RELIABLE_RELIABILITY_QOS
        self.writer_qos.durability().kind = fastdds.TRANSIENT_LOCAL_DURABILITY_QOS


        # self.writer_qos.history_depth = 20

        # history_qos = fastdds.HistoryQosPolicy()
        # history_qos.kind = fastdds.KEEP_ALL_HISTORY_QOS
        # # # history_qos.depth = 20
        # self.writer_qos.history(history_qos)

        self.writer = self.publisher.create_datawriter(self.topic, self.writer_qos, self.listener)


    def write(self):
        data = MultiCameraData.ImageData()
        lenna_image = cv2.imread('/home/juyiii/Desktop/Lenna.png')
        # 获取lenna_image的宽和高
        print(f"lenna image shape: ", lenna_image.shape)
        data.width(lenna_image.shape[1])
        data.height(lenna_image.shape[0])

        # data.width(100)
        # data.height(100)
        # 写入data 全0
        # data.data([255] * 480 * 640)
        # len = np.array(data.data()).tolist()

        # # 生成随机的灰度图数据并赋值到 data.data
        # 生成随机的灰度图数据，并确保数据格式正确
        # random_data = np.random.randint(0, 256, size=(100 * 100 * 3), dtype=np.uint8)
        # JPEG compression parameters
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 50]
        compressed_lens = []  # List to store compressed lengths for each camera
        _, encoded_image = cv2.imencode('.jpg', lenna_image, encode_param)
        # print("compressed_image: {compressed_size}".format(compressed_size=encoded_image.size))
        encoded_image_bytes = encoded_image.tobytes()
        data.data(encoded_image_bytes)  # 使用 bytes 直接传输
        # data.data(encoded_image.tolist())
        # 将 NumPy 数组转换为 list 格式
        # data.data = list(random_data)  # 直接转换为 Python 列表
        decoded_image = cv2.imdecode(encoded_image, cv2.IMREAD_COLOR)
        print(f"Decoded image shape: {decoded_image.shape}")

        self.writer.write(data)
        # print("Sending {message} : {index}".format(message=data.message(), index=data.index()))
        print("Published {width} , {height}, {datasize}".format(width=data.width(), height=data.height(),
                                                                datasize = data.data().size()))

        self.index = self.index + 1


    def wait_discovery(self) :
        self._cvDiscovery.acquire()
        print ("Writer is waiting discovery...")
        self._cvDiscovery.wait_for(lambda : self._matched_reader != 0)
        self._cvDiscovery.release()
        print("Writer discovery finished...")


    def run(self):
        self.wait_discovery()
        for x in range(50):
            time.sleep(0.01)
            self.write()
        self.delete()


    def delete(self):
        factory = fastdds.DomainParticipantFactory.get_instance()
        self.participant.delete_contained_entities()
        factory.delete_participant(self.participant)


if __name__ == '__main__':
    print('Starting publisher.')
    writer = Writer()
    writer.run()
    exit()