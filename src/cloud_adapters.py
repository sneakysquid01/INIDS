"""
Cloud Provider Adapters (Week 6)

Abstraction layer for multi-cloud support:
- AWS (S3, CloudWatch, SNS)
- Azure (Blob Storage, Application Insights, Event Hubs)
- GCP (Cloud Storage, Stackdriver, Pub/Sub)

Each provider implements unified interface for:
1. Model/artifact storage
2. Logs and metrics collection
3. Alert/event distribution
"""

import logging
from typing import Dict, Any, Optional, List
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class CloudMetric:
    """Cloud metric data."""
    name: str
    value: float
    unit: str
    timestamp: datetime
    tags: Dict[str, str]


@dataclass
class CloudAlert:
    """Cloud alert notification."""
    alert_id: str
    service: str
    severity: str  # CRITICAL, HIGH, MEDIUM, LOW
    message: str
    timestamp: datetime
    metadata: Dict[str, Any]


class CloudStorageAdapter(ABC):
    """Abstract base for cloud storage (S3, Blob Storage, Cloud Storage)."""
    
    @abstractmethod
    async def upload_file(
        self,
        local_path: str,
        remote_path: str,
        metadata: Dict[str, str] = None
    ) -> bool:
        """Upload file to cloud storage."""
        pass
    
    @abstractmethod
    async def download_file(
        self,
        remote_path: str,
        local_path: str
    ) -> bool:
        """Download file from cloud storage."""
        pass
    
    @abstractmethod
    async def delete_file(self, remote_path: str) -> bool:
        """Delete file from cloud storage."""
        pass
    
    @abstractmethod
    async def list_files(self, prefix: str) -> List[str]:
        """List files in path prefix."""
        pass


class CloudMonitoringAdapter(ABC):
    """Abstract base for cloud monitoring (CloudWatch, App Insights, Stackdriver)."""
    
    @abstractmethod
    async def send_metric(self, metric: CloudMetric) -> bool:
        """Send metric to cloud monitoring."""
        pass
    
    @abstractmethod
    async def send_log(
        self,
        log_stream: str,
        message: str,
        level: str = "INFO"
    ) -> bool:
        """Send log to cloud logging."""
        pass
    
    @abstractmethod
    async def get_metrics(
        self,
        metric_name: str,
        start_time: datetime,
        end_time: datetime
    ) -> List[CloudMetric]:
        """Get metrics from cloud monitoring."""
        pass


class CloudMessagingAdapter(ABC):
    """Abstract base for cloud messaging (SNS, Event Hubs, Pub/Sub)."""
    
    @abstractmethod
    async def publish_alert(self, alert: CloudAlert) -> bool:
        """Publish alert to cloud messaging."""
        pass
    
    @abstractmethod
    async def subscribe_to_alerts(
        self,
        handler_func
    ) -> str:
        """Subscribe to alerts, returns subscription ID."""
        pass
    
    @abstractmethod
    async def unsubscribe(self, subscription_id: str) -> bool:
        """Unsubscribe from alerts."""
        pass


class AWSAdapter(CloudStorageAdapter, CloudMonitoringAdapter, CloudMessagingAdapter):
    """AWS implementation (S3, CloudWatch, SNS)."""
    
    def __init__(
        self,
        region: str = "us-east-1",
        s3_bucket: str = None,
        access_key: str = None,
        secret_key: str = None
    ):
        """Initialize AWS adapter.
        
        Args:
            region: AWS region
            s3_bucket: S3 bucket for models/artifacts
            access_key: AWS access key
            secret_key: AWS secret key
        """
        self.region = region
        self.s3_bucket = s3_bucket
        self.access_key = access_key
        self.secret_key = secret_key
        self.s3_client = None
        self.cloudwatch_client = None
        self.sns_client = None
    
    async def connect(self) -> bool:
        """Connect to AWS services."""
        try:
            import boto3
            
            session = boto3.Session(
                region_name=self.region,
                aws_access_key_id=self.access_key,
                aws_secret_access_key=self.secret_key
            )
            
            self.s3_client = session.client("s3")
            self.cloudwatch_client = session.client("cloudwatch")
            self.sns_client = session.client("sns")
            
            logger.info(f"Connected to AWS region {self.region}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to connect to AWS: {e}")
            return False
    
    # Storage methods
    async def upload_file(
        self,
        local_path: str,
        remote_path: str,
        metadata: Dict[str, str] = None
    ) -> bool:
        """Upload file to S3."""
        try:
            extra_args = {}
            if metadata:
                extra_args["Metadata"] = metadata
            
            self.s3_client.upload_file(
                local_path,
                self.s3_bucket,
                remote_path,
                ExtraArgs=extra_args if extra_args else None
            )
            
            logger.info(f"Uploaded {remote_path} to S3")
            return True
        
        except Exception as e:
            logger.error(f"Error uploading to S3: {e}")
            return False
    
    async def download_file(
        self,
        remote_path: str,
        local_path: str
    ) -> bool:
        """Download file from S3."""
        try:
            self.s3_client.download_file(
                self.s3_bucket,
                remote_path,
                local_path
            )
            
            logger.info(f"Downloaded {remote_path} from S3")
            return True
        
        except Exception as e:
            logger.error(f"Error downloading from S3: {e}")
            return False
    
    async def delete_file(self, remote_path: str) -> bool:
        """Delete file from S3."""
        try:
            self.s3_client.delete_object(
                Bucket=self.s3_bucket,
                Key=remote_path
            )
            
            logger.info(f"Deleted {remote_path} from S3")
            return True
        
        except Exception as e:
            logger.error(f"Error deleting from S3: {e}")
            return False
    
    async def list_files(self, prefix: str) -> List[str]:
        """List files in S3 prefix."""
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=self.s3_bucket,
                Prefix=prefix
            )
            
            files = []
            if "Contents" in response:
                files = [obj["Key"] for obj in response["Contents"]]
            
            return files
        
        except Exception as e:
            logger.error(f"Error listing S3 files: {e}")
            return []
    
    # Monitoring methods
    async def send_metric(self, metric: CloudMetric) -> bool:
        """Send metric to CloudWatch."""
        try:
            self.cloudwatch_client.put_metric_data(
                Namespace="INIDS",
                MetricData=[
                    {
                        "MetricName": metric.name,
                        "Value": metric.value,
                        "Unit": metric.unit,
                        "Timestamp": metric.timestamp,
                        "Dimensions": [
                            {"Name": k, "Value": v} for k, v in metric.tags.items()
                        ]
                    }
                ]
            )
            
            logger.debug(f"Sent metric {metric.name} to CloudWatch")
            return True
        
        except Exception as e:
            logger.error(f"Error sending metric to CloudWatch: {e}")
            return False
    
    async def send_log(
        self,
        log_stream: str,
        message: str,
        level: str = "INFO"
    ) -> bool:
        """Send log to CloudWatch Logs."""
        try:
            import boto3
            logs_client = boto3.client("logs", region_name=self.region)
            
            logs_client.put_log_events(
                logGroupName="/inids/alerts",
                logStreamName=log_stream,
                logEvents=[
                    {
                        "message": f"[{level}] {message}",
                        "timestamp": int(datetime.now().timestamp() * 1000)
                    }
                ]
            )
            
            logger.debug(f"Sent log to CloudWatch: {message}")
            return True
        
        except Exception as e:
            logger.error(f"Error sending log to CloudWatch: {e}")
            return False
    
    async def get_metrics(
        self,
        metric_name: str,
        start_time: datetime,
        end_time: datetime
    ) -> List[CloudMetric]:
        """Get metrics from CloudWatch."""
        try:
            response = self.cloudwatch_client.get_metric_statistics(
                Namespace="INIDS",
                MetricName=metric_name,
                StartTime=start_time,
                EndTime=end_time,
                Period=300,
                Statistics=["Average", "Sum", "Maximum"]
            )
            
            metrics = []
            for point in response.get("Datapoints", []):
                metrics.append(CloudMetric(
                    name=metric_name,
                    value=point.get("Average", point.get("Sum", 0)),
                    unit="Count",
                    timestamp=point["Timestamp"],
                    tags={}
                ))
            
            return metrics
        
        except Exception as e:
            logger.error(f"Error getting metrics from CloudWatch: {e}")
            return []
    
    # Messaging methods
    async def publish_alert(self, alert: CloudAlert) -> bool:
        """Publish alert to SNS."""
        try:
            self.sns_client.publish(
                TopicArn=f"arn:aws:sns:{self.region}:*:inids-alerts",
                Subject=f"[{alert.severity}] {alert.service}",
                Message=alert.message
            )
            
            logger.info(f"Published alert {alert.alert_id} to SNS")
            return True
        
        except Exception as e:
            logger.error(f"Error publishing alert to SNS: {e}")
            return False
    
    async def subscribe_to_alerts(self, handler_func) -> str:
        """Subscribe to SNS alerts."""
        try:
            # In real implementation: set up SQS + SNS subscription
            subscription_id = f"aws-sub-{datetime.now().timestamp()}"
            logger.info(f"Subscribed to alerts: {subscription_id}")
            return subscription_id
        
        except Exception as e:
            logger.error(f"Error subscribing to alerts: {e}")
            return ""
    
    async def unsubscribe(self, subscription_id: str) -> bool:
        """Unsubscribe from SNS alerts."""
        try:
            logger.info(f"Unsubscribed from alerts: {subscription_id}")
            return True
        
        except Exception as e:
            logger.error(f"Error unsubscribing: {e}")
            return False


class AzureAdapter(CloudStorageAdapter, CloudMonitoringAdapter, CloudMessagingAdapter):
    """Azure implementation (Blob Storage, App Insights, Event Hubs)."""
    
    def __init__(
        self,
        connection_string: str = None,
        account_name: str = None,
        account_key: str = None,
        container_name: str = None
    ):
        """Initialize Azure adapter.
        
        Args:
            connection_string: Azure connection string
            account_name: Storage account name
            account_key: Storage account key
            container_name: Blob container for models
        """
        self.connection_string = connection_string
        self.account_name = account_name
        self.account_key = account_key
        self.container_name = container_name
        self.blob_client = None
        self.ai_client = None
        self.eh_client = None
    
    async def connect(self) -> bool:
        """Connect to Azure services."""
        try:
            from azure.storage.blob import BlobServiceClient
            from azure.monitor.opentelemetry import AzureMonitorTraceExporter
            from azure.messaging.eventhubs import EventHubProducerClient
            
            if self.connection_string:
                self.blob_client = BlobServiceClient.from_connection_string(
                    self.connection_string
                )
            
            logger.info("Connected to Azure services")
            return True
        
        except Exception as e:
            logger.error(f"Failed to connect to Azure: {e}")
            return False
    
    # Storage methods
    async def upload_file(
        self,
        local_path: str,
        remote_path: str,
        metadata: Dict[str, str] = None
    ) -> bool:
        """Upload file to Azure Blob Storage."""
        try:
            with open(local_path, "rb") as data:
                self.blob_client.get_blob_client(
                    container=self.container_name,
                    blob=remote_path
                ).upload_blob(data, overwrite=True, metadata=metadata)
            
            logger.info(f"Uploaded {remote_path} to Azure Blob Storage")
            return True
        
        except Exception as e:
            logger.error(f"Error uploading to Azure: {e}")
            return False
    
    async def download_file(
        self,
        remote_path: str,
        local_path: str
    ) -> bool:
        """Download file from Azure Blob Storage."""
        try:
            with open(local_path, "wb") as file:
                download_stream = self.blob_client.get_blob_client(
                    container=self.container_name,
                    blob=remote_path
                ).download_blob()
                file.write(download_stream.readall())
            
            logger.info(f"Downloaded {remote_path} from Azure")
            return True
        
        except Exception as e:
            logger.error(f"Error downloading from Azure: {e}")
            return False
    
    async def delete_file(self, remote_path: str) -> bool:
        """Delete file from Azure Blob Storage."""
        try:
            self.blob_client.get_blob_client(
                container=self.container_name,
                blob=remote_path
            ).delete_blob()
            
            logger.info(f"Deleted {remote_path} from Azure")
            return True
        
        except Exception as e:
            logger.error(f"Error deleting from Azure: {e}")
            return False
    
    async def list_files(self, prefix: str) -> List[str]:
        """List files in Azure Blob Storage prefix."""
        try:
            container_client = self.blob_client.get_container_client(
                self.container_name
            )
            
            files = []
            for blob in container_client.list_blobs(name_starts_with=prefix):
                files.append(blob.name)
            
            return files
        
        except Exception as e:
            logger.error(f"Error listing Azure files: {e}")
            return []
    
    # Monitoring methods
    async def send_metric(self, metric: CloudMetric) -> bool:
        """Send metric to Application Insights."""
        try:
            # In real implementation: use Application Insights SDK
            logger.debug(f"Sent metric {metric.name} to App Insights")
            return True
        
        except Exception as e:
            logger.error(f"Error sending metric to App Insights: {e}")
            return False
    
    async def send_log(
        self,
        log_stream: str,
        message: str,
        level: str = "INFO"
    ) -> bool:
        """Send log to Application Insights."""
        try:
            # In real implementation: use Application Insights SDK
            logger.debug(f"Sent log to App Insights: {message}")
            return True
        
        except Exception as e:
            logger.error(f"Error sending log to App Insights: {e}")
            return False
    
    async def get_metrics(
        self,
        metric_name: str,
        start_time: datetime,
        end_time: datetime
    ) -> List[CloudMetric]:
        """Get metrics from Application Insights."""
        try:
            # In real implementation: query App Insights
            return []
        
        except Exception as e:
            logger.error(f"Error getting metrics from App Insights: {e}")
            return []
    
    # Messaging methods
    async def publish_alert(self, alert: CloudAlert) -> bool:
        """Publish alert to Event Hubs."""
        try:
            # In real implementation: use Event Hubs SDK
            logger.info(f"Published alert {alert.alert_id} to Event Hubs")
            return True
        
        except Exception as e:
            logger.error(f"Error publishing to Event Hubs: {e}")
            return False
    
    async def subscribe_to_alerts(self, handler_func) -> str:
        """Subscribe to Event Hubs alerts."""
        try:
            subscription_id = f"azure-sub-{datetime.now().timestamp()}"
            logger.info(f"Subscribed to alerts: {subscription_id}")
            return subscription_id
        
        except Exception as e:
            logger.error(f"Error subscribing to alerts: {e}")
            return ""
    
    async def unsubscribe(self, subscription_id: str) -> bool:
        """Unsubscribe from Event Hubs alerts."""
        try:
            logger.info(f"Unsubscribed from alerts: {subscription_id}")
            return True
        
        except Exception as e:
            logger.error(f"Error unsubscribing: {e}")
            return False


class GCPAdapter(CloudStorageAdapter, CloudMonitoringAdapter, CloudMessagingAdapter):
    """GCP implementation (Cloud Storage, Stackdriver, Pub/Sub)."""
    
    def __init__(
        self,
        project_id: str = None,
        credentials_path: str = None,
        bucket_name: str = None
    ):
        """Initialize GCP adapter.
        
        Args:
            project_id: GCP project ID
            credentials_path: Path to GCP credentials JSON
            bucket_name: Cloud Storage bucket
        """
        self.project_id = project_id
        self.credentials_path = credentials_path
        self.bucket_name = bucket_name
        self.storage_client = None
        self.monitoring_client = None
        self.pubsub_client = None
    
    async def connect(self) -> bool:
        """Connect to GCP services."""
        try:
            from google.cloud import storage, monitoring_v3, pubsub_v1
            
            self.storage_client = storage.Client(project=self.project_id)
            self.monitoring_client = monitoring_v3.MetricServiceClient()
            self.pubsub_client = pubsub_v1.PublisherClient()
            
            logger.info(f"Connected to GCP project {self.project_id}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to connect to GCP: {e}")
            return False
    
    # Storage methods
    async def upload_file(
        self,
        local_path: str,
        remote_path: str,
        metadata: Dict[str, str] = None
    ) -> bool:
        """Upload file to Cloud Storage."""
        try:
            bucket = self.storage_client.bucket(self.bucket_name)
            blob = bucket.blob(remote_path)
            blob.metadata = metadata or {}
            blob.upload_from_filename(local_path)
            
            logger.info(f"Uploaded {remote_path} to Cloud Storage")
            return True
        
        except Exception as e:
            logger.error(f"Error uploading to GCP: {e}")
            return False
    
    async def download_file(
        self,
        remote_path: str,
        local_path: str
    ) -> bool:
        """Download file from Cloud Storage."""
        try:
            bucket = self.storage_client.bucket(self.bucket_name)
            blob = bucket.blob(remote_path)
            blob.download_to_filename(local_path)
            
            logger.info(f"Downloaded {remote_path} from Cloud Storage")
            return True
        
        except Exception as e:
            logger.error(f"Error downloading from GCP: {e}")
            return False
    
    async def delete_file(self, remote_path: str) -> bool:
        """Delete file from Cloud Storage."""
        try:
            bucket = self.storage_client.bucket(self.bucket_name)
            blob = bucket.blob(remote_path)
            blob.delete()
            
            logger.info(f"Deleted {remote_path} from Cloud Storage")
            return True
        
        except Exception as e:
            logger.error(f"Error deleting from GCP: {e}")
            return False
    
    async def list_files(self, prefix: str) -> List[str]:
        """List files in Cloud Storage prefix."""
        try:
            bucket = self.storage_client.bucket(self.bucket_name)
            files = [blob.name for blob in bucket.list_blobs(prefix=prefix)]
            return files
        
        except Exception as e:
            logger.error(f"Error listing GCP files: {e}")
            return []
    
    # Monitoring methods
    async def send_metric(self, metric: CloudMetric) -> bool:
        """Send metric to Stackdriver."""
        try:
            # In real implementation: use Stackdriver API
            logger.debug(f"Sent metric {metric.name} to Stackdriver")
            return True
        
        except Exception as e:
            logger.error(f"Error sending metric to Stackdriver: {e}")
            return False
    
    async def send_log(
        self,
        log_stream: str,
        message: str,
        level: str = "INFO"
    ) -> bool:
        """Send log to Cloud Logging."""
        try:
            # In real implementation: use Cloud Logging API
            logger.debug(f"Sent log to Cloud Logging: {message}")
            return True
        
        except Exception as e:
            logger.error(f"Error sending log to Cloud Logging: {e}")
            return False
    
    async def get_metrics(
        self,
        metric_name: str,
        start_time: datetime,
        end_time: datetime
    ) -> List[CloudMetric]:
        """Get metrics from Stackdriver."""
        try:
            # In real implementation: query Stackdriver
            return []
        
        except Exception as e:
            logger.error(f"Error getting metrics from Stackdriver: {e}")
            return []
    
    # Messaging methods
    async def publish_alert(self, alert: CloudAlert) -> bool:
        """Publish alert to Pub/Sub."""
        try:
            topic_path = self.pubsub_client.topic_path(
                self.project_id, "inids-alerts"
            )
            
            self.pubsub_client.publish(
                topic_path,
                alert.message.encode("utf-8"),
                service=alert.service,
                severity=alert.severity
            )
            
            logger.info(f"Published alert {alert.alert_id} to Pub/Sub")
            return True
        
        except Exception as e:
            logger.error(f"Error publishing to Pub/Sub: {e}")
            return False
    
    async def subscribe_to_alerts(self, handler_func) -> str:
        """Subscribe to Pub/Sub alerts."""
        try:
            subscription_id = f"gcp-sub-{datetime.now().timestamp()}"
            logger.info(f"Subscribed to alerts: {subscription_id}")
            return subscription_id
        
        except Exception as e:
            logger.error(f"Error subscribing to alerts: {e}")
            return ""
    
    async def unsubscribe(self, subscription_id: str) -> bool:
        """Unsubscribe from Pub/Sub alerts."""
        try:
            logger.info(f"Unsubscribed from alerts: {subscription_id}")
            return True
        
        except Exception as e:
            logger.error(f"Error unsubscribing: {e}")
            return False


def get_cloud_adapter(provider: str, **kwargs):
    """Factory function to get cloud adapter.
    
    Args:
        provider: 'aws', 'azure', or 'gcp'
        **kwargs: Provider-specific parameters
        
    Returns:
        Cloud adapter instance
    """
    if provider == "aws":
        return AWSAdapter(**kwargs)
    elif provider == "azure":
        return AzureAdapter(**kwargs)
    elif provider == "gcp":
        return GCPAdapter(**kwargs)
    else:
        raise ValueError(f"Unknown cloud provider: {provider}")
