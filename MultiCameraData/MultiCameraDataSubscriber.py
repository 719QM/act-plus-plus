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
ImageData Subscriber
"""
import signal

import fastdds
import MultiCameraData
import cv2
import numpy as np
import time

DESCRIPTION = """ImageData Subscriber example for Fast DDS python bindings"""
USAGE = ('python3 MultiCameraDataSubscriber.py')

# To capture ctrl+C
def signal_handler(sig, frame):
    print('Interrupted!')

class ReaderListener(fastdds.DataReaderListener):


    def __init__(self):
        super().__init__()
        self.fps = 0
        self.last_message_time = None
        self.index = 0
        self.image_100 = None
        self.image_110 = None


    def on_subscription_matched(self, datareader, info) :
        if (0 < info.current_count_change) :
            print ("Subscriber matched publisher {}".format(info.last_publication_handle))
        else :
            print ("Subscriber unmatched publisher {}".format(info.last_publication_handle))


    def on_data_available(self, reader):
        info = fastdds.SampleInfo()
        data = MultiCameraData.ImageData()
        reader.take_next_sample(data, info)
        self.index = self.index + 1

        # print("Received {message} : {index}".format(message=data.message(), index=data.index()))
        # print("Type of data.data:", type(data.data))
        print(f"index: ", self.index, end=" ")
        # 获取接收到的数据
        raw_camera_100 = data.camera100()
        raw_camera_110 = data.camera110()
        # raw_data = data.data()
        # print(f"Type of raw_data: {type(raw_data)}, Length of raw_data: {len(raw_data) if raw_data else 0}")
        if raw_camera_100 is None or len(raw_camera_100) == 0:
            print("Error: Camera 100 receive empty data.")
            return
        else:
            raw_camera_100_bytes = bytes(raw_camera_100)
            image_100_array = np.frombuffer(raw_camera_100_bytes, dtype=np.uint8)
            print(f"image_100_array size: ", image_100_array.size, end=" ")
            self.image_100 = cv2.imdecode(image_100_array, cv2.IMREAD_COLOR)
            print("image 100 shape: ", self.image_100.shape, end=" ")

        if raw_camera_110 is None or len(raw_camera_110) == 0:
            print("Error: Camera 110 receive empty data.")
            return
        else:
            raw_camera_110_bytes = bytes(raw_camera_110)
            image_110_array = np.frombuffer(raw_camera_110_bytes, dtype=np.uint8)
            print(f"image_110_array size: ", image_110_array.size, end=" ")
            self.image_110 = cv2.imdecode(image_110_array, cv2.IMREAD_COLOR)
            print("image 110 shape: ", self.image_110.shape, end=" ")


        # print("Received {width} , {height}, {datasize}".format(width=data.width(), height=data.height(),
        #                                                        datasize=len(data.data())), end=" ")
        # image_array = np.array(data.data(), dtype=np.uint8)
        # frame = image_array.reshape((data.height(), data.width(), 3))

        # 显示图像
        cv2.imshow('Image 100', self.image_100)
        cv2.imshow('Image 110', self.image_110)
        cv2.waitKey(1)
        current_time = time.time()
        if self.last_message_time is not None:
            self.fps = 1 / (current_time - self.last_message_time)

        self.last_message_time = current_time
        print("FPS: ", self.fps)


class Reader:


    def __init__(self):
        factory = fastdds.DomainParticipantFactory.get_instance()
        self.participant_qos = fastdds.DomainParticipantQos()
        # transit_qos = fastdds.TransportConfigQos()
        # transit_qos.listen_socket_buffer_size = 12582912
        # self.participant_qos.transport(transit_qos)


        factory.get_default_participant_qos(self.participant_qos)
        self.participant = factory.create_participant(0, self.participant_qos)

        self.topic_data_type = MultiCameraData.ImageDataPubSubType()
        self.topic_data_type.set_name("MultiCameraData")
        self.type_support = fastdds.TypeSupport(self.topic_data_type)
        self.participant.register_type(self.type_support)

        self.topic_qos = fastdds.TopicQos()
        self.participant.get_default_topic_qos(self.topic_qos)
        self.topic = self.participant.create_topic("MultiCameraDataTopic", self.topic_data_type.get_name(), self.topic_qos)

        self.subscriber_qos = fastdds.SubscriberQos()
        self.participant.get_default_subscriber_qos(self.subscriber_qos)
        self.subscriber = self.participant.create_subscriber(self.subscriber_qos)

        self.listener = ReaderListener()
        self.reader_qos = fastdds.DataReaderQos()
        self.subscriber.get_default_datareader_qos(self.reader_qos)

        # self.reader_qos.history = fastdds.KEEP_ALL_HISTORY_QOS
        self.reader_qos.history().kind = fastdds.KEEP_ALL_HISTORY_QOS
        self.reader_qos.reliability().kind = fastdds.BEST_EFFORT_RELIABILITY_QOS
        self.reader_qos.durability().kind = fastdds.TRANSIENT_LOCAL_DURABILITY_QOS
        # self.reader_qos.liveliness = fastdds.AUTOMATIC_LIVELINESS_QOS
        # self.reader_qos.lifespan = 10

        # history_qos = fastdds.HistoryQosPolicy()
        # history_qos.kind = fastdds.KEEP_ALL_HISTORY_QOS
        # # # history_qos.depth = 50
        # self.reader_qos.history(history_qos)

        self.reader = self.subscriber.create_datareader(self.topic, self.reader_qos, self.listener)


    def delete(self):
        factory = fastdds.DomainParticipantFactory.get_instance()
        self.participant.delete_contained_entities()
        factory.delete_participant(self.participant)


    def run(self):
        signal.signal(signal.SIGINT, signal_handler)
        print('Press Ctrl+C to stop')
        signal.pause()
        self.delete()


if __name__ == '__main__':
    print('Creating subscriber.')
    reader = Reader()
    reader.run()
    exit()