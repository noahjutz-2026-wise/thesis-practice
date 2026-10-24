import ctypes

import iceoryx2 as iox2


def main():
    node = (
        iox2.NodeBuilder.new()
        .name(iox2.NodeName.new("Subscriber"))
        .create(iox2.ServiceType.Ipc)
    )
    service = (
        node.service_builder(iox2.ServiceName.new("CamService"))
        .publish_subscribe(ctypes.c_float)
        .open_or_create()
    )

    subscriber = service.subscriber_builder().create()

    print("ready")

    while True:
        node.wait(iox2.Duration.from_secs(1))
        sample = subscriber.receive()
        if sample is None:
            continue

        print(sample.payload().contents)


if __name__ == "__main__":
    main()
